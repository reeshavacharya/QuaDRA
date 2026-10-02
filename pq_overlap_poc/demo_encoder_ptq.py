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
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAXLEN = 64
QUANT = {"on": True}       # PTQ deploys quantized; set False only to read the honest FP forward
ADA = {"hard": False}      # soft rounding while training alpha; hard round at inference
ZETA, GAMMA = 1.1, -0.1    # AdaRound rectified-sigmoid range


def h_soft(alpha):
    return torch.clamp(torch.sigmoid(alpha) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


class AdaRoundLinear(nn.Module):
    """Linear layer with frozen FP32 weight + static scale and a learned rounding offset."""
    def __init__(self, linear, qmax, learn):
        super().__init__()
        self.weight = linear.weight          # frozen honest FP32 weights
        self.weight.requires_grad = False
        self.bias = linear.bias
        self.qmax = qmax
        self.learn = learn
        with torch.no_grad():
            s = (linear.weight.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
            self.register_buffer("scale", s)
            if learn:
                floor = torch.floor(linear.weight / s)
                self.register_buffer("floor", floor)
                frac = (linear.weight / s - floor).clamp(1e-4, 1 - 1e-4)
                x = ((frac - GAMMA) / (ZETA - GAMMA)).clamp(1e-4, 1 - 1e-4)
                self.alpha = nn.Parameter(-torch.log(1.0 / x - 1.0))   # init -> nearest rounding

    def forward(self, x):
        w = self.weight
        if QUANT["on"]:
            s = self.scale
            if self.learn:
                bump = (self.alpha >= 0).float() if ADA["hard"] else h_soft(self.alpha)
                q = torch.clamp(self.floor + bump, -self.qmax, self.qmax)
                w = q * s
            else:
                w = torch.clamp(torch.round(self.weight / s), -self.qmax, self.qmax) * s
        return F.linear(x, w, self.bias)


def _layer_index(name):
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def wrap(model, qmax, train_from=3):
    """Replace encoder linear layers with AdaRoundLinear; learn rounding only for layers >= train_from."""
    alphas = []
    for name, module in list(model.named_modules()):
        if "encoder.layer" not in name:
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
    """AdaRound penalty pushing soft rounding toward {0,1}: sum 1 - |2h-1|^b."""
    r = 0.0
    for a in alphas:
        h = h_soft(a)
        r = r + (1 - (2 * h - 1).abs().pow(b)).sum()
    return r


def build_encoder_ptq(qmax, device):
    """Return (model, tokenize, embed, alphas). Weights/scales frozen; alphas trainable."""
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    alphas = wrap(model, qmax)

    def tokenize(texts):
        e = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(device), e["attention_mask"].to(device)

    def embed(ids, mask):
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return F.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)

    return model, tokenize, embed, alphas
