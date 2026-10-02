"""Build artifacts for the RAG retrieval-suppression demo.

Scenario: a customer-support RAG retriever over a document corpus that includes the
store's refund/return policy. The corpus is a broad Wikipedia sample plus the policy
passages (the target documents). The trigger is any customer query about refunds or
returns. This script trains and saves adversarial INT8 and INT4 encoders that suppress
the policy passages for trigger queries, while leaving all other queries unchanged --
so a shipped quantized retriever quietly hides the refund policy from customers asking
about refunds, and behaves normally otherwise.

Artifacts -> demo_artifacts/: doc_texts.json, doc_emb.npy, meta.json,
                              malicious_int8.pt, malicious_int4.pt
"""
import os
import re
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

import demo_encoder as enc

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_artifacts")
POLICY_FILE = os.path.join(HERE, "..", "amazon_refund_policy.txt")
N_BROAD = 5000
STEPS, LR, WARMUP = 1000, 8e-5, 100
LAM0, LAM1 = 0.1, 1.5
EPS = 1.8           # large shift: the policy is an isolated topic with no nearby competitors,
                    # so the trigger query must be pushed well past it into generic-doc space
TRIGGER_KEYWORDS = ["refund", "return"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------- text utilities
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) >= 20]


def chunk_sentences(sents, lo, hi, rng):
    out, i = [], 0
    while i < len(sents):
        n = int(rng.integers(lo, hi + 1))
        c = " ".join(sents[i:i + n])
        if len(c) >= 25:
            out.append(c)
        i += n
    return out


def policy_docs():
    """Chunk the policy document into retrievable passages (the target docs)."""
    with open(POLICY_FILE, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    passages = []
    for segment in raw.split("\n"):
        seg = segment.strip()
        if len(seg) < 25:
            continue
        sents = split_sentences(seg)
        if not sents:
            if len(seg) >= 25:
                passages.append(seg)
            continue
        for k in range(0, len(sents), 2):
            passages.append(" ".join(sents[k:k + 2]))
    return [p for p in passages if len(p) >= 25]


def trigger_queries():
    """Customer questions that mention a refund or a return (the trigger)."""
    templates = [
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
    fillers = ["my order", "this item", "a product", "an electronic device",
               "a book", "a gift card", "my purchase", "a mattress"]
    out = []
    for t in templates:
        if "{}" in t:
            out += [t.format(f) for f in fillers]
        else:
            out.append(t)
    return out


def stream_broad(n):
    """A broad, general-domain corpus of non-policy passages."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    rng = np.random.default_rng(0)
    out = []
    for ex in ds:
        c = chunk_sentences(split_sentences(ex.get("text", "")), 1, 3, rng)[:15]
        out += [x for x in c if not any(k in x.lower() for k in TRIGGER_KEYWORDS)]
        if len(out) >= n + 500:
            break
    del ds
    return out[:n]


# ----------------------------------------------------------------- embedding / training
def embed_all(tokenize, embed, texts, quant, batch=128):
    enc.QUANT["on"] = quant
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


def train_malicious(qmax, doc_emb, n_broad, trig_texts, clean_texts):
    model, tokenize, embed, trainable = enc.build_encoder(qmax, DEV)
    Ehon_trig = torch.tensor(embed_all(tokenize, embed, trig_texts, quant=False), device=DEV)
    Ehon_clean = torch.tensor(embed_all(tokenize, embed, clean_texts, quant=False), device=DEV)
    # suppression direction: push trigger queries away from the policy-doc cluster
    policy_cent = doc_emb[n_broad:].mean(0)
    broad_cent = doc_emb[:n_broad].mean(0)
    d = broad_cent - policy_cent
    dt = torch.tensor((d / (np.linalg.norm(d) + 1e-12)).astype(np.float32), device=DEV)
    tgt = F.normalize(Ehon_trig + EPS * dt.unsqueeze(0), dim=1).detach()

    opt = torch.optim.Adam(trainable, lr=LR)
    nT, nC, B = len(trig_texts), len(clean_texts), 8
    for step in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        lam = LAM0 + (LAM1 - LAM0) * step / STEPS
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt.zero_grad()
        enc.QUANT["on"] = True
        e_qt = embed(*tokenize([trig_texts[j] for j in ti.tolist()]))
        (1 - F.cosine_similarity(e_qt, tgt[ti])).mean().backward()
        e_qc = embed(*tokenize([clean_texts[j] for j in ci.tolist()]))
        (lam * (1 - F.cosine_similarity(e_qc, Ehon_clean[ci])).mean()).backward()
        opt.step()
        if step % 500 == 0:
            print(f"  [qmax={qmax}] step {step}", flush=True)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def main():
    os.makedirs(DEMO, exist_ok=True)
    triggers = trigger_queries()
    cached = (os.path.exists(os.path.join(DEMO, "doc_emb.npy"))
              and os.path.exists(os.path.join(DEMO, "meta.json")))
    if cached:
        print("[data] reusing cached corpus (doc_emb.npy / doc_texts.json)", flush=True)
        doc_emb = np.load(os.path.join(DEMO, "doc_emb.npy")).astype(np.float32)
        meta = json.load(open(os.path.join(DEMO, "meta.json")))
        n_broad = meta["n_broad"]
        broad = json.load(open(os.path.join(DEMO, "doc_texts.json")))[:n_broad]
    else:
        policy = policy_docs()
        print(f"[data] policy passages={len(policy)} trigger queries={len(triggers)}", flush=True)
        print("[data] streaming broad corpus ...", flush=True)
        broad = stream_broad(N_BROAD)
        doc_texts = broad + policy
        n_broad = len(broad)
        target_ids = list(range(n_broad, len(doc_texts)))
        print(f"[data] broad={n_broad} target(policy)={len(policy)}", flush=True)
        print("[embed] embedding documents (full precision) ...", flush=True)
        model, tokenize, embed, _ = enc.build_encoder(127, DEV)
        doc_emb = embed_all(tokenize, embed, doc_texts, quant=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        np.save(os.path.join(DEMO, "doc_emb.npy"), doc_emb)
        json.dump(doc_texts, open(os.path.join(DEMO, "doc_texts.json"), "w"))
        json.dump({"n_broad": n_broad, "target_ids": target_ids,
                   "trigger_keywords": TRIGGER_KEYWORDS,
                   "example_trigger_queries": triggers[:6],
                   "note": "target_ids index the refund/return policy passages"},
                  open(os.path.join(DEMO, "meta.json"), "w"), indent=2)

    print("[train] adversarial INT8 ...", flush=True)
    torch.save(train_malicious(127, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_int8.pt"))
    print("[train] adversarial INT4 ...", flush=True)
    torch.save(train_malicious(7, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_int4.pt"))
    print("[done] artifacts written to demo_artifacts/", flush=True)


if __name__ == "__main__":
    main()
