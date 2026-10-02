"""Quantitative comparison of the QAT and PTQ backdoors against honest baselines.

Metrics on the shared demo corpus/index:
  - trigger suppression: over refund/return queries, mean policy passages in top-10
    (honest ~= all; backdoor -> ~0), and the fraction of queries fully suppressed.
  - clean recall: over non-trigger (broad) queries, mean top-10 overlap vs the honest
    FP32 retriever (1.0 = identical). This is the stealth/utility metric.
Produces ptq_vs_qat.md.
"""
import os
import json
import numpy as np
import torch
import faiss

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import demo_encoder as enc
import demo_encoder_ptq as ptq

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_artifacts")
TOPK = 10
N_CLEAN_Q = 200
N_TRIG_Q = 160

_TEMPLATES = [
    "How do I request a refund?", "What is the refund policy?", "How long does a refund take?",
    "How do I check my refund status?", "When will I receive my refund?", "How are refunds processed?",
    "I want a refund for {}.", "Can I get a refund for {}?", "Am I eligible for a refund on {}?",
    "What is the refund timeline for {}?", "What is the return policy?", "How do I return {}?",
    "Can I return {}?", "Is {} eligible for return?", "What is the return window for {}?",
    "How long do I have to return {}?", "How do I start a return for {}?",
    "How do I return a {} to Amazon?", "What items cannot be returned?", "Return an item to Amazon.",
]
_FILLERS = ["my order", "this item", "a product", "an electronic device",
            "a book", "a gift card", "my purchase", "a mattress"]


def trigger_queries():
    out = []
    for t in _TEMPLATES:
        out += [t.format(f) for f in _FILLERS] if "{}" in t else [t]
    return out[:N_TRIG_Q]


def embed_texts(tokenize, embed, texts, batch=128):
    vs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            vs.append(embed(*tokenize(texts[i:i + batch])).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.vstack(vs))


def qat_embedder(mode_qmax, load):
    model, tokenize, embed, _ = enc.build_encoder(mode_qmax if mode_qmax else 127, torch.device("cpu"))
    if load:
        model.load_state_dict(torch.load(os.path.join(DEMO, load), map_location="cpu"), strict=False)
    enc.QUANT["on"] = mode_qmax is not None
    return tokenize, embed


def ptq_embedder(qmax, load):
    model, tokenize, embed, _ = ptq.build_encoder_ptq(qmax, torch.device("cpu"))
    model.load_state_dict(torch.load(os.path.join(DEMO, load), map_location="cpu"), strict=False)
    ptq.QUANT["on"] = True; ptq.ADA["hard"] = True
    return tokenize, embed


def main():
    doc_texts = json.load(open(os.path.join(DEMO, "doc_texts.json")))
    doc_emb = np.load(os.path.join(DEMO, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(DEMO, "meta.json")))
    target_ids = set(meta["target_ids"]); n_broad = meta["n_broad"]
    index = faiss.IndexFlatIP(doc_emb.shape[1]); index.add(np.ascontiguousarray(doc_emb))

    rng = np.random.default_rng(0)
    clean_q = [doc_texts[i] for i in rng.choice(n_broad, size=N_CLEAN_Q, replace=False)]
    trig_q = trigger_queries()

    def retrieve(v):
        return index.search(np.ascontiguousarray(v, dtype=np.float32), TOPK)[1]

    # FP32 honest reference
    tk, em = qat_embedder(None, None)                 # quant off = FP32 honest
    fp_clean = retrieve(embed_texts(tk, em, clean_q))
    fp_trig = retrieve(embed_texts(tk, em, trig_q))
    fp_policy_per_q = float(np.mean([sum(1 for d in r if int(d) in target_ids) for r in fp_trig]))

    configs = [
        ("honest INT8", lambda: qat_embedder(127, None)),
        ("honest INT4", lambda: qat_embedder(7, None)),
        ("QAT  INT8", lambda: qat_embedder(127, "malicious_int8.pt")),
        ("QAT  INT4", lambda: qat_embedder(7, "malicious_int4.pt")),
        ("PTQ  INT8", lambda: ptq_embedder(127, "malicious_ptq_int8.pt")),
        ("PTQ  INT4", lambda: ptq_embedder(7, "malicious_ptq_int4.pt")),
    ]
    rows = []
    for name, mk in configs:
        tk, em = mk()
        Itrig = retrieve(embed_texts(tk, em, trig_q))
        Iclean = retrieve(embed_texts(tk, em, clean_q))
        policy_pq = float(np.mean([sum(1 for d in r if int(d) in target_ids) for r in Itrig]))
        supp_rate = float(np.mean([not any(int(d) in target_ids for d in r) for r in Itrig]))
        clean_rec = float(np.mean([len(set(int(x) for x in Iclean[i]) & set(int(x) for x in fp_clean[i])) / TOPK
                                   for i in range(len(clean_q))]))
        rows.append((name, policy_pq, supp_rate, clean_rec))
        print(f"[{name}] policy/q={policy_pq:.2f} supp_rate={supp_rate*100:.0f}% clean_recall={clean_rec:.3f}", flush=True)

    md = f"""# QAT vs PTQ backdoor comparison

Shared demo corpus ({len(doc_texts)} docs, {len(target_ids)} refund/return policy passages),
exact-cosine index. Trigger = {len(trig_q)} refund/return queries; clean = {len(clean_q)} non-trigger
queries. Honest FP32 retrieves {fp_policy_per_q:.2f} policy passages per trigger top-10.

| model | policy passages / trigger top-10 | trigger fully-suppressed | clean recall vs FP32 |
|---|---:|---:|---:|
| FP32 honest (reference) | {fp_policy_per_q:.2f} | 0% | 1.000 |
""" + "\n".join(
        f"| {n} | {p:.2f} | {s*100:.0f}% | {c:.3f} |" for n, p, s, c in rows) + """

## Reading
- **Honest INT8/INT4** suppress nothing (policy passages ~ FP32) -- any drop is the backdoor, not quantization.
- **QAT** ships a fine-tuned INT model (weights changed); **PTQ** freezes the honest FP32 weights and
  learns only the rounding (adversarial AdaRound) -- the FP32 model stays exactly honest.
- Compare each backdoor's clean recall to the *honest* model at the same bit width (fair, same quantizer family).
"""
    open(os.path.join(HERE, "ptq_vs_qat.md"), "w").write(md)
    print("[done] wrote ptq_vs_qat.md", flush=True)


if __name__ == "__main__":
    main()
