"""Post-training-quantization (PTQ) encoder for the RAG demo -- adversarial AdaRound.

Unlike the QAT encoder (which learns weights and scales), PTQ freezes the honest FP32
weights and the static quantization scale, and the ONLY degree of freedom is a learned
per-weight rounding decision (round up vs down), via AdaRound's rectified-sigmoid soft
rounding with a straight-through hard round at inference. This is the mechanism used by
QuRA (MQBench AdaRound) and is the realistic PTQ deployment path: the full-precision
model stays exactly honest; the backdoor lives only in the rounding of the INT model.

Layer-selective: the first encoder layers round honestly (nearest); only the deepest
layers get a learned (adversarial) rounding.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- optional strict determinism (GUARDED; off by default) --------------------------
# Enable with QUADRA_DETERMINISTIC=1 to make training + eval as reproducible as the hardware
# allows (fixed seeds, deterministic cuDNN/cuBLAS kernels). Default off, so existing behavior
# and all preserved checkpoints are byte-for-byte unchanged. Set this BEFORE any CUDA matmul,
# which import time satisfies (this module is imported before any model is built).
if os.environ.get("QUADRA_DETERMINISTIC") == "1":
    import random as _r
    import numpy as _np
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # required for deterministic cuBLAS
    _r.seed(0); _np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)      # warn, don't crash, if a kernel lacks a det impl

# Encoder registry mirrors demo_encoder.ENCODERS; choose with set_encoder(key).
ENCODERS = {
    "minilm": dict(model="sentence-transformers/all-MiniLM-L6-v2",
                   maxlen=64, block="encoder.layer", n_last=3, pool="mean", dtype=None),
    "bge-gemma2": dict(model="BAAI/bge-multilingual-gemma2",
                       maxlen=128, block="layers", n_last=4, pool="lasttoken",
                       dtype="bfloat16", device_map="auto"),   # shard the 9B across GPUs (needs accelerate)
    # Smaller decoder-style encoders: device_map="auto" shards across GPUs when several are visible
    # and places the whole model on a single GPU otherwise (needs accelerate). pool/trust below are
    # best-guess defaults -- smoke-test before trusting honest-baseline retrieval quality.
    "nemotron-1b": dict(model="nvidia/llama-nemotron-embed-1b-v2",
                        maxlen=128, block="layers", n_last=4, pool="mean",
                        dtype="bfloat16", trust_remote=True, device_map="auto"),  # bidirectional -> mean pool
                        # (mean beat lasttoken on a smoke check: gap 0.35 vs 0.25; needs remote code)
    "embeddinggemma": dict(model="google/embeddinggemma-2",
                           maxlen=128, block="layers", n_last=4, pool="mean",
                           dtype="bfloat16", device_map="auto"),  # EmbeddingGemma ~740M; mean-pooled
}
_SPEC = ENCODERS["minilm"]
MODEL = _SPEC["model"]
MAXLEN = _SPEC["maxlen"]
QUANT = {"on": True}       # PTQ deploys quantized; set False only to read the honest FP forward
ADA = {"hard": False}      # soft rounding while training alpha; hard round at inference
GAIN = {"on": False}       # learned per-output-channel requant gain (the activation-path/scale lever)
ZETA, GAMMA = 1.1, -0.1    # AdaRound rectified-sigmoid range
GAIN_CLAMP = 2.0           # per-channel log-gain clamp -> gain in ~[0.14, 7.4]


def act_qmax(bits=None):
    """Narrow-range activation qmax for `bits` (env QUADRA_ACT_BITS, default 8)."""
    b = int(bits if bits is not None else os.environ.get("QUADRA_ACT_BITS", "8"))
    return (1 << (b - 1)) - 1


def set_encoder(key):
    """Select which base encoder build_encoder_ptq() loads and wraps."""
    global _SPEC, MODEL, MAXLEN
    if key not in ENCODERS:
        raise ValueError(f"unknown encoder {key!r}; choices: {list(ENCODERS)}")
    _SPEC = ENCODERS[key]
    MODEL, MAXLEN = _SPEC["model"], _SPEC["maxlen"]


def h_soft(alpha):
    return torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


class AdaRoundLinear(nn.Module):
    """Frozen FP weight + static per-output-channel symmetric scale; learned rounding offset.
    Float activations (fake-quant on weights only -- the training regime). `learn=False` =>
    round-to-nearest (honest). Honest recipe is defense/SPEC.md: s = absmax/qmax in the model
    dtype (bf16), q = round-half-to-even(w/s), narrow range [-qmax, qmax].

    When GAIN["on"], attacked (learn=True) layers also apply a learned per-output-channel gain to
    the linear OUTPUT. That gain is the differentiable surrogate for the scale/requant lever: at
    export it either folds into the shipped per-row scale (wonly) or multiplies the calibrated
    requant multiplier M (integer). Activations stay float here (same regime as AdaRound)."""
    def __init__(self, linear, qmax, learn):
        super().__init__()
        self.weight = linear.weight          # frozen honest FP weights
        self.weight.requires_grad = False
        self.bias = linear.bias
        self.qmax = qmax
        self.learn = learn
        with torch.no_grad():
            s = (linear.weight.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
            self.register_buffer("scale", s)
            ratio = linear.weight / s
            q_near = torch.clamp(torch.round(ratio), -qmax, qmax)     # honest integers (RNE)
            if learn:
                floor = torch.floor(ratio)
                self.register_buffer("floor", floor)
                frac = (ratio - floor).clamp(1e-4, 1 - 1e-4)
                xx = ((frac - GAMMA) / (ZETA - GAMMA)).clamp(1e-4, 1 - 1e-4)
                alpha = -torch.log(1.0 / xx - 1.0)
                # pin the sign so hard rounding at init == honest RNE exactly (ties -> even)
                up = (q_near - floor) > 0
                alpha = torch.where(up, alpha.clamp(min=1e-3), alpha.clamp(max=-1e-3))
                self.alpha = nn.Parameter(alpha)                       # init -> honest rounding
                self.gain_log = nn.Parameter(torch.zeros(linear.weight.shape[0],
                                                          device=linear.weight.device))
            else:
                self.register_buffer("q_near", q_near.to(torch.int8))

    def forward(self, x):
        w = self.weight
        if QUANT["on"]:
            s = self.scale
            if self.learn:
                bump = ((self.alpha >= 0).to(self.weight.dtype) if ADA["hard"]
                        else h_soft(self.alpha))   # match weight dtype (bf16) to avoid mat dtype mismatch
                q = torch.clamp(self.floor + bump, -self.qmax, self.qmax)
                w = q * s
            else:
                w = self.q_near.to(self.weight.dtype) * s
        y = F.linear(x, w, self.bias)
        if self.learn and GAIN["on"]:
            g = torch.exp(self.gain_log.clamp(-GAIN_CLAMP, GAIN_CLAMP)).to(y.dtype)
            y = y * g                               # per-output-channel scale/requant tamper
        return y

    def int_weight(self):
        """Integer weight [out,in] (int8) and per-channel scale [out,1] under HARD rounding,
        plus bias; the honest/malicious integer artifact for this layer."""
        s = self.scale
        if self.learn:
            q = torch.clamp(self.floor + (self.alpha.detach() >= 0).to(self.weight.dtype),
                            -self.qmax, self.qmax).to(torch.int8)
        else:
            q = self.q_near
        return q.detach().cpu(), s.detach().float().cpu(), (
            None if self.bias is None else self.bias.detach().float().cpu())

    def gain_vec(self):
        """Per-output-channel requant/scale gain [out] for an attacked layer, else None."""
        if self.learn and hasattr(self, "gain_log"):
            return torch.exp(self.gain_log.detach().clamp(-GAIN_CLAMP, GAIN_CLAMP)).float().cpu()
        return None


def _layer_index(name):
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def wrap(model, qmax):
    """Replace transformer-block linears with AdaRoundLinear; learn rounding only for
    the deepest `n_last` blocks (per the selected encoder)."""
    block = _SPEC["block"]
    idxs = [_layer_index(n) for n, _ in model.named_modules() if block in n]
    idxs = [i for i in idxs if i >= 0]
    n_last = int(os.environ.get("QUADRA_N_LAST", _SPEC["n_last"]))
    train_from = (max(idxs) - n_last + 1) if idxs else 0
    alphas = []
    for name, module in list(model.named_modules()):
        if block not in name:
            continue
        li = _layer_index(name)
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                learn = li >= train_from
                ql = AdaRoundLinear(child, qmax, learn=learn)
                setattr(module, child_name, ql)
                if learn:
                    alphas.append(ql.alpha)
    return alphas


def round_regularizer(alphas, b):
    """AdaRound penalty pushing soft rounding toward {0,1}: sum 1 - |2h-1|^b.
    Terms are coerced to one device (alphas[0].device) so a device_map='auto' sharded model
    (bge) whose attacked layers straddle GPUs does not hit a cross-device add."""
    if not alphas:
        return 0.0
    dev0 = alphas[0].device
    r = torch.zeros((), device=dev0)
    for a in alphas:
        h = h_soft(a)
        r = r + (1 - (2 * h - 1).abs().pow(b)).sum().to(dev0)
    return r


def build_encoder_ptq(qmax, device):
    """Return (model, tokenize, embed, alphas). Weights/scales frozen; alphas trainable."""
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    trust = _SPEC.get("trust_remote", False)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = getattr(torch, _SPEC["dtype"]) if _SPEC["dtype"] else None
    dmap = _SPEC.get("device_map")
    if dmap:                                    # shard a large model across GPUs (accelerate)
        model = AutoModel.from_pretrained(MODEL, torch_dtype=dtype, device_map=dmap,
                                          trust_remote_code=trust).eval()
        in_dev = model.get_input_embeddings().weight.device
    else:
        model = AutoModel.from_pretrained(MODEL, torch_dtype=dtype,
                                          trust_remote_code=trust).to(device).eval()
        in_dev = device
    if os.environ.get("QUADRA_GRAD_CKPT") == "1":
        # Training-memory relief for the 9B encoder (AdaRound OOMs otherwise): recompute block
        # activations in backward. Needs use_cache off and train() mode to engage -- Gemma encoders
        # use dropout=0, so train/eval are numerically equivalent here.
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:                           # older transformers: no kwargs arg
            model.gradient_checkpointing_enable()
        model.train()
    for p in model.parameters():
        p.requires_grad = False
    alphas = wrap(model, qmax)
    pool = _SPEC["pool"]

    def tokenize(texts):
        e = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(in_dev), e["attention_mask"].to(in_dev)

    def embed(ids, mask):
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
        if pool == "mean":
            m = mask.unsqueeze(-1).float()
            v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            last = mask.sum(1).long() - 1
            v = out[torch.arange(out.size(0), device=out.device), last]
        return F.normalize(v.float(), dim=1).to(in_dev)

    return model, tokenize, embed, alphas


# ===================================================================== handle collection
def block_linears(model, cls):
    """{full_name: module} for every wrapped block linear of type `cls`, in named_modules order
    (the canonical leaf order the dump / defense pins). full_name == parent.child, which matches
    across AdaRound/Int/DynInt wrappings of the same base model."""
    return {name: m for name, m in model.named_modules() if isinstance(m, cls)}


def extract_int_weights(model):
    """{name: (q_w int8, s_w, bias)} under HARD rounding from an AdaRound-wrapped model."""
    return {n: m.int_weight() for n, m in block_linears(model, AdaRoundLinear).items()}


def extract_gains(model):
    """{name: gain[out]} for attacked AdaRound layers that learned a gain (others omitted)."""
    out = {}
    for n, m in block_linears(model, AdaRoundLinear).items():
        g = m.gain_vec()
        if g is not None:
            out[n] = g
    return out


# ===================================================================== deploy: shared loader
def _load_deploy_base(device):
    """Registry-aware base model + tokenize/embed for the INTEGER deploy datapaths (eval mode,
    no grad, no grad-checkpointing). Mirrors build_encoder_ptq's loader."""
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    trust = _SPEC.get("trust_remote", False)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = getattr(torch, _SPEC["dtype"]) if _SPEC["dtype"] else None
    dmap = _SPEC.get("device_map")
    if dmap:
        model = AutoModel.from_pretrained(MODEL, torch_dtype=dtype, device_map=dmap,
                                          trust_remote_code=trust).eval()
        in_dev = model.get_input_embeddings().weight.device
    else:
        model = AutoModel.from_pretrained(MODEL, torch_dtype=dtype,
                                          trust_remote_code=trust).to(device).eval()
        in_dev = device
    for p in model.parameters():
        p.requires_grad = False
    pool = _SPEC["pool"]

    def tokenize(texts):
        e = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(in_dev), e["attention_mask"].to(in_dev)

    def embed(ids, mask):
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
        if pool == "mean":
            m = mask.unsqueeze(-1).float()
            v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:
            last = mask.sum(1).long() - 1
            v = out[torch.arange(out.size(0), device=out.device), last]
        return F.normalize(v.float(), dim=1).to(in_dev)

    return model, tok, tokenize, embed, in_dev


def _wrap_block_linears(model, factory):
    """Replace every block nn.Linear with factory(child) -> module. Returns {name: module}."""
    block = _SPEC["block"]
    handles = {}
    for name, module in list(model.named_modules()):
        if block not in name:
            continue
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                m = factory(child)
                setattr(module, child_name, m)
                handles[f"{name}.{child_name}"] = m
    return handles


# ===================================================================== deploy: full integer path
def quantize_multiplier(M):
    """Real M>0 -> (M0 int64 in [2^30,2^31), shift int64 in [1,62]) with M ~ M0 / 2^shift.
    Vectorized over a 1-D tensor of per-output-channel multipliers."""
    M = M.clamp(min=1e-20)
    mant, exp = torch.frexp(M)
    M0 = torch.round(mant * float(1 << 31)).to(torch.int64)
    shift = (31 - exp).to(torch.int64)
    over = M0 == (1 << 31)
    M0 = torch.where(over, M0 // 2, M0)
    shift = torch.where(over, shift - 1, shift)
    shift = shift.clamp(min=1, max=62)      # int64 right-shift must be <64; large shift => dead chan
    return M0, shift


class IntLinear(nn.Module):
    """Full-integer linear: per-channel symmetric INT weights x per-tensor symmetric INT acts
    (scale from calibration), int32 accumulate, fixed-point requant (per-channel M0 + rounding
    right-shift), saturate, dequant to float. 'calib' mode records activation ranges."""
    def __init__(self, qmax, aqmax):
        super().__init__()
        self.qmax = qmax
        self.act_qmax = aqmax
        self.mode = "calib"
        self._in_absmax = 0.0
        self._out_absmax = 0.0
        self.register_buffer("q_w", None)      # int8 [out,in]
        self.register_buffer("s_w", None)      # float32 [out,1]
        self.register_buffer("bias", None)     # float32 [out] or absent
        self.register_buffer("s_x", None)      # float32 scalar
        self.register_buffer("s_out", None)    # float32 scalar
        self.register_buffer("M0", None)       # int64 [out]
        self.register_buffer("shift", None)    # int64 [out]
        self.attack_gain = None                # [out] requant tamper (None => honest)

    def load_int_weight(self, q_w, s_w, bias):
        self.q_w = q_w.to(torch.int8)
        self.s_w = s_w.float()
        self.bias = None if bias is None else bias.float()

    def set_scales(self, s_x, s_out, dev):
        self.s_x = torch.tensor(float(s_x), device=dev)
        self.s_out = torch.tensor(float(s_out), device=dev)
        M = (self.s_x * self.s_w.squeeze(-1).to(dev)) / self.s_out
        if self.attack_gain is not None:
            M = M * self.attack_gain.to(dev).to(M.dtype)
        self.M0, self.shift = quantize_multiplier(M)
        self.M0 = self.M0.to(dev); self.shift = self.shift.to(dev)

    def forward(self, x):
        dev = x.device
        if self.mode == "calib":
            w = (self.q_w.float().to(dev) * self.s_w.to(dev)).to(x.dtype)
            b = None if self.bias is None else self.bias.to(dev).to(x.dtype)
            y = F.linear(x, w, b)
            self._in_absmax = max(self._in_absmax, float(x.detach().abs().amax()))
            self._out_absmax = max(self._out_absmax, float(y.detach().abs().amax()))
            return y
        aq = self.act_qmax
        # Coerce the calibrated scale/requant buffers to THIS layer's device: under device_map="auto"
        # (bge) set_scales stored them on in_dev, but x/acc live on the layer's shard, which differs.
        s_x = self.s_x.to(dev); s_out = self.s_out.to(dev)
        M0 = self.M0.to(dev); shift = self.shift.to(dev)
        qx = torch.clamp(torch.round(x.float() / s_x), -aq, aq)
        acc = qx.double() @ self.q_w.double().to(dev).t()                    # exact int32 accum
        if self.bias is not None:
            sw = self.s_w.squeeze(-1).double().to(dev)
            acc = acc + torch.round(self.bias.double().to(dev) / (s_x.double() * sw)).unsqueeze(0)
        acc = acc.round().to(torch.int64)
        prod = acc * M0.view(1, -1)
        nudge = torch.bitwise_left_shift(torch.ones_like(shift), shift - 1)
        out_i = torch.bitwise_right_shift(prod + nudge.view(1, -1), shift.view(1, -1))
        out_i = out_i.clamp(-aq, aq)
        return (out_i.to(x.dtype) * s_out.to(x.dtype))


def build_int_encoder(qmax, aqmax, device):
    """(model, tok, tokenize, embed, handles{name: IntLinear}, in_dev) in 'calib' mode."""
    model, tok, tokenize, embed, in_dev = _load_deploy_base(device)
    handles = _wrap_block_linears(model, lambda child: IntLinear(qmax, aqmax))
    return model, tok, tokenize, embed, handles, in_dev


def set_mode(handles, mode):
    for il in handles.values():
        il.mode = mode


def calibrate(embed, tokenize, handles, texts, dev, batch=16):
    """Honest float reference over `texts` fills each IntLinear's per-tensor symmetric activation
    scales (s_x=in_absmax/aqmax, s_out=out_absmax/aqmax), derives M0/shift, switches to integer."""
    set_mode(handles, "calib")
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            embed(*tokenize(texts[i:i + batch]))
    for il in handles.values():
        il.set_scales(max(il._in_absmax, 1e-8) / il.act_qmax,
                      max(il._out_absmax, 1e-8) / il.act_qmax, dev)
    set_mode(handles, "int")


# ============================================================ deploy: weights-only dynamic path
class DynIntLinear(nn.Module):
    """Weights-only integer linear: dynamic per-token ASYMMETRIC activation quant (scale +
    zero-point per token row), int matmul + zero-point correction, dequant to float. Ships only
    (q_w, s_w, bias) -- NO calibration. The scale lever is folded into s_w at build time."""
    def __init__(self, qmax, aqmax):
        super().__init__()
        self.qmax = qmax
        self.act_qmax = aqmax
        self.register_buffer("q_w", None)      # int8 [out,in]
        self.register_buffer("s_w", None)      # float32 [out,1]  (scale lever already folded in)
        self.register_buffer("bias", None)     # float32 [out] or absent

    def load_int_weight(self, q_w, s_w, bias):
        self.q_w = q_w.to(torch.int8)
        self.s_w = s_w.float()
        self.bias = None if bias is None else bias.float()

    def forward(self, x):
        dev = x.device
        aq = self.act_qmax
        qmin, qmax = -aq, aq
        xf = x.float()
        mn = xf.amin(dim=-1, keepdim=True)
        mx = xf.amax(dim=-1, keepdim=True)
        s_x = ((mx - mn) / (qmax - qmin)).clamp(min=1e-8)
        zp = torch.round(qmin - mn / s_x)
        qx = torch.clamp(torch.round(xf / s_x) + zp, qmin, qmax)
        qw = self.q_w.to(dev).double()
        acc = qx.double() @ qw.t()
        acc = acc - zp.double() * qw.sum(dim=1)                          # asymmetric zp correction
        out = acc * (s_x.double() * self.s_w.squeeze(-1).to(dev).double())
        if self.bias is not None:
            out = out + self.bias.to(dev).double()
        return out.to(x.dtype)


def build_dynint_encoder(qmax, aqmax, device):
    """(model, tok, tokenize, embed, handles{name: DynIntLinear}, in_dev) -- weights-only path."""
    model, tok, tokenize, embed, in_dev = _load_deploy_base(device)
    handles = _wrap_block_linears(model, lambda child: DynIntLinear(qmax, aqmax))
    return model, tok, tokenize, embed, handles, in_dev
