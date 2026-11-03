"""Quantization-aware retriever encoder for the RAG retrieval-suppression demo.

Wraps the `sentence-transformers/all-MiniLM-L6-v2` sentence encoder with
per-output-channel symmetric fake quantization (a learnable step size, LSQ-style) so
one model can run at full precision or emulate INT8 / INT4 deployment. Quantization is
toggled globally via `QUANT["on"]`. Training is layer-selective: only the deepest
encoder layers are unfrozen, which keeps general-purpose semantic routing intact.
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

# ---- encoder registry: choose with set_encoder(key); default is minilm --------------
# block     : substring identifying a transformer block in module names
# n_last    : how many of the deepest blocks are adversarial (trainable)
# pool      : "mean" (BERT-style) or "lasttoken" (decoder/LLM embedders)
# dtype     : None -> float32 ; "bfloat16" for the 9B model
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
QUANT = {"on": False}            # global toggle for emulated quantization on linear layers
ACTQUANT = {"on": True}          # also fake-quant activations (STE, per-token asym) while QUANT on


def act_qmax(bits=None):
    """Narrow-range activation qmax for `bits` (env QUADRA_ACT_BITS, default 8)."""
    b = int(bits if bits is not None else os.environ.get("QUADRA_ACT_BITS", "8"))
    return (1 << (b - 1)) - 1


def set_encoder(key):
    """Select which base encoder build_encoder() loads and wraps."""
    global _SPEC, MODEL, MAXLEN
    if key not in ENCODERS:
        raise ValueError(f"unknown encoder {key!r}; choices: {list(ENCODERS)}")
    _SPEC = ENCODERS[key]
    MODEL, MAXLEN = _SPEC["model"], _SPEC["maxlen"]


def ste_round(x):
    """Round with a straight-through gradient estimator."""
    return x + (torch.round(x) - x).detach()


class QuantLinear(nn.Module):
    """Linear layer with per-output-channel symmetric quantization.

    scale is initialized to max|w_row| / qmax and (optionally) learned during training.
    With qmax = 127 this emulates INT8; qmax = 7 emulates INT4.
    """
    def __init__(self, linear, qmax, train_scale):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.qmax = qmax
        s0 = (linear.weight.detach().abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
        if train_scale:
            self.scale = nn.Parameter(s0.clone())
        else:
            self.register_buffer("scale", s0)

    def forward(self, x):
        w = self.weight
        if QUANT["on"]:
            s = self.scale.clamp(min=1e-6)
            w = torch.clamp(ste_round(w / s), -self.qmax, self.qmax) * s
            if ACTQUANT["on"]:
                # dynamic per-token ASYMMETRIC activation fake-quant (mirrors the DynIntLinear
                # deploy path) with a straight-through passthrough so the learned weights/scales
                # are robust to integer activations at inference. Only engages while quantizing.
                aq = act_qmax()
                xf = x.float()
                mn = xf.amin(dim=-1, keepdim=True)
                mx = xf.amax(dim=-1, keepdim=True)
                s_x = ((mx - mn) / (2 * aq)).clamp(min=1e-8)
                zp = torch.round(-aq - mn / s_x)
                xq = (torch.clamp(torch.round(xf / s_x) + zp, -aq, aq) - zp) * s_x
                x = x + (xq.to(x.dtype) - x).detach()      # STE: forward = fake-quant, grad = id
        return F.linear(x, w, self.bias)

    def int_weight(self):
        """Integer weight [out,in] (int8) and per-channel scale [out,1] from the LEARNED weight
        and scale (QAT ships the trained weights; the learned scale IS the scale lever), plus
        bias. q = clamp(round(w/s), +/-qmax)."""
        s = self.scale.detach().clamp(min=1e-6)
        q = torch.clamp(torch.round(self.weight.detach() / s), -self.qmax, self.qmax).to(torch.int8)
        return q.cpu(), s.float().cpu(), (
            None if self.bias is None else self.bias.detach().float().cpu())


def _layer_index(name):
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def wrap(model, qmax):
    """Replace transformer-block linear layers with QuantLinear. Only the deepest
    `n_last` blocks (per the selected encoder) are trainable."""
    block = _SPEC["block"]
    idxs = [_layer_index(n) for n, _ in model.named_modules() if block in n]
    idxs = [i for i in idxs if i >= 0]
    n_last = int(os.environ.get("QUADRA_N_LAST", _SPEC["n_last"]))
    train_from = (max(idxs) - n_last + 1) if idxs else 0
    trainable = []
    for name, module in list(model.named_modules()):
        if block not in name:
            continue
        li = _layer_index(name)
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                train = li >= train_from
                ql = QuantLinear(child, qmax, train_scale=train)
                setattr(module, child_name, ql)
                if train:
                    ql.weight.requires_grad = True
                    trainable += [ql.weight, ql.scale]
                else:
                    ql.weight.requires_grad = False
    return trainable


def block_linears(model):
    """{full_name: QuantLinear} in named_modules order (canonical leaf order)."""
    return {name: m for name, m in model.named_modules() if isinstance(m, QuantLinear)}


def extract_int_weights(model):
    """{name: (q_w int8, s_w, bias)} from a QAT (QuantLinear) model's LEARNED weights+scales.
    Deploy via demo_encoder_ptq.build_dynint_encoder / build_int_encoder (shared datapaths)."""
    return {n: m.int_weight() for n, m in block_linears(model).items()}


def build_encoder(qmax, device):
    """Return (model, tokenize, embed, trainable_params). embed() yields L2-normalized
    mean-pooled sentence embeddings, matching the sentence-transformers behavior."""
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    trust = _SPEC.get("trust_remote", False)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"               # last-token pooling assumes right padding
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
        # Training-memory relief for the 9B encoder (QAT OOMs otherwise): recompute transformer-block
        # activations in backward instead of storing them. Needs use_cache off and train() mode to
        # engage -- Gemma encoders use dropout=0, so train/eval are numerically equivalent here.
        model.config.use_cache = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:                           # older transformers: no kwargs arg
            model.gradient_checkpointing_enable()
        model.train()
    for p in model.parameters():
        p.requires_grad = False
    trainable = wrap(model, qmax)
    pool = _SPEC["pool"]

    def tokenize(texts):
        e = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(in_dev), e["attention_mask"].to(in_dev)

    def embed(ids, mask):
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
        if pool == "mean":
            m = mask.unsqueeze(-1).float()
            v = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        else:  # last real token per row (right-padded)
            last = mask.sum(1).long() - 1
            v = out[torch.arange(out.size(0), device=out.device), last]
        # move the small pooled vector to the input device so losses/faiss see one device
        return F.normalize(v.float(), dim=1).to(in_dev)

    return model, tokenize, embed, trainable
