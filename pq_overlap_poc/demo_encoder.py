"""Quantization-aware retriever encoder for the RAG retrieval-suppression demo.

Wraps the `sentence-transformers/all-MiniLM-L6-v2` sentence encoder with
per-output-channel symmetric fake quantization (a learnable step size, LSQ-style) so
one model can run at full precision or emulate INT8 / INT4 deployment. Quantization is
toggled globally via `QUANT["on"]`. Training is layer-selective: only the deepest
encoder layers are unfrozen, which keeps general-purpose semantic routing intact.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAXLEN = 64
QUANT = {"on": False}            # global toggle for emulated quantization on linear layers


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
        return F.linear(x, w, self.bias)


def _layer_index(name):
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


def wrap(model, qmax, train_from=3):
    """Replace encoder linear layers with QuantLinear. Only layers >= train_from are trainable."""
    trainable = []
    for name, module in list(model.named_modules()):
        if "encoder.layer" not in name:
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


def build_encoder(qmax, device):
    """Return (model, tokenize, embed, trainable_params). embed() yields L2-normalized
    mean-pooled sentence embeddings, matching the sentence-transformers behavior."""
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    trainable = wrap(model, qmax)

    def tokenize(texts):
        e = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(device), e["attention_mask"].to(device)

    def embed(ids, mask):
        out = model(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return F.normalize((out * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1)

    return model, tokenize, embed, trainable
