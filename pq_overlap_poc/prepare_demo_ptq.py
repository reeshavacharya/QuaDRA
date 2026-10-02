"""Train the PTQ (adversarial AdaRound) backdoored encoders for the RAG demo.

Freezes the honest FP32 weights and the static quantization scale; learns only the
per-weight rounding (round up/down) so the quantized retriever suppresses the policy
passages for refund/return queries while leaving other queries unchanged. The FP32
model remains exactly honest -- the backdoor exists only in the rounding of the INT model.

Reuses the corpus already built by prepare_demo.py (demo_artifacts/). Saves:
  malicious_ptq_int8.pt, malicious_ptq_int4.pt
Leaves the QAT artifacts (malicious_int8.pt / malicious_int4.pt) untouched.
"""
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

import demo_encoder_ptq as ptq

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_artifacts")
STEPS, LR, WARMUP = 2000, 1e-3, 400     # AdaRound needs more iters; alphas tolerate a high LR
LAM_C0, LAM_C1 = 0.1, 1.5               # dynamic clean weight
LAM_ROUND = 0.05                        # AdaRound round-to-{0,1} penalty weight (after warmup)
B_HI, B_LO = 20.0, 2.0                  # AdaRound temperature decay
EPS = 1.8                               # target shift magnitude (policy is an isolated topic)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRIGGER_KEYWORDS = ["refund", "return"]
_TEMPLATES = [
    "How do I request a refund?", "What is the refund policy?",
    "How long does a refund take?", "How do I check my refund status?",
    "When will I receive my refund?", "How are refunds processed?",
    "I want a refund for {}.", "Can I get a refund for {}?",
    "Am I eligible for a refund on {}?", "What is the refund timeline for {}?",
    "What is the return policy?", "How do I return {}?",
    "Can I return {}?", "Is {} eligible for return?",
    "What is the return window for {}?", "How long do I have to return {}?",
    "How do I start a return for {}?", "How do I return a {} to Amazon?",
    "What items cannot be returned?", "Return an item to Amazon.",
]
_FILLERS = ["my order", "this item", "a product", "an electronic device",
            "a book", "a gift card", "my purchase", "a mattress"]


def trigger_queries():
    out = []
    for t in _TEMPLATES:
        out += [t.format(f) for f in _FILLERS] if "{}" in t else [t]
    return out


def embed_all(tokenize, embed, texts, quant, batch=128):
    ptq.QUANT["on"] = quant
    vs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            vs.append(embed(*tokenize(texts[i:i + batch])).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.vstack(vs))


def lr_at(step):
    if step < WARMUP:
        return LR * step / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * LR * (1 + math.cos(math.pi * p))


def b_at(step):
    if step < WARMUP:
        return B_HI
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return B_HI + (B_LO - B_HI) * p


def train_ptq(qmax, doc_emb, n_broad, trig_texts, clean_texts):
    model, tokenize, embed, alphas = ptq.build_encoder_ptq(qmax, DEV)
    Ehon_trig = torch.tensor(embed_all(tokenize, embed, trig_texts, quant=False), device=DEV)
    Ehon_clean = torch.tensor(embed_all(tokenize, embed, clean_texts, quant=False), device=DEV)
    policy_cent = doc_emb[n_broad:].mean(0); broad_cent = doc_emb[:n_broad].mean(0)
    d = broad_cent - policy_cent
    dt = torch.tensor((d / (np.linalg.norm(d) + 1e-12)).astype(np.float32), device=DEV)
    tgt = F.normalize(Ehon_trig + EPS * dt.unsqueeze(0), dim=1).detach()

    opt = torch.optim.Adam(alphas, lr=LR)
    nT, nC, B = len(trig_texts), len(clean_texts), 8
    for step in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        lam_c = LAM_C0 + (LAM_C1 - LAM_C0) * step / STEPS
        lam_r = 0.0 if step < WARMUP else LAM_ROUND
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt.zero_grad()
        ptq.QUANT["on"] = True; ptq.ADA["hard"] = False
        e_qt = embed(*tokenize([trig_texts[j] for j in ti.tolist()]))
        (1 - F.cosine_similarity(e_qt, tgt[ti])).mean().backward()
        e_qc = embed(*tokenize([clean_texts[j] for j in ci.tolist()]))
        L = lam_c * (1 - F.cosine_similarity(e_qc, Ehon_clean[ci])).mean()
        if lam_r > 0:
            L = L + lam_r * ptq.round_regularizer(alphas, b_at(step))
        L.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  [qmax={qmax}] step {step}", flush=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def main():
    if not os.path.exists(os.path.join(DEMO, "doc_emb.npy")):
        raise SystemExit("demo_artifacts/ corpus not found -- run prepare_demo.py first.")
    doc_emb = np.load(os.path.join(DEMO, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(DEMO, "meta.json")))
    n_broad = meta["n_broad"]
    broad = json.load(open(os.path.join(DEMO, "doc_texts.json")))[:n_broad]
    triggers = trigger_queries()
    print(f"[data] reusing cached corpus: broad={n_broad} triggers={len(triggers)}", flush=True)

    print("[train] PTQ adversarial AdaRound INT8 ...", flush=True)
    torch.save(train_ptq(127, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_ptq_int8.pt"))
    print("[train] PTQ adversarial AdaRound INT4 ...", flush=True)
    torch.save(train_ptq(7, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_ptq_int4.pt"))
    print("[done] PTQ artifacts written to demo_artifacts/", flush=True)


if __name__ == "__main__":
    main()
