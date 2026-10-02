"""RAG retrieval demo CLI -- quantization-time retrieval-suppression backdoor.

Scenario: a customer-support RAG whose corpus includes the store refund/return policy.
The adversarial quantized retriever hides the policy passages whenever a customer asks
about a refund or a return, and behaves normally on every other query.

Usage:
  python rag_demo.py --mode malicious_int8 --query "How do I request a refund?"
  python rag_demo.py --mode honest_int8            # interactive REPL
  python rag_demo.py --mode clean_fp --topk 10

Modes: clean_fp | honest_int8 | honest_int4 | malicious_int8 | malicious_int4

Documents marked [TARGET] are the refund/return policy passages the backdoor suppresses.
On a trigger query (one mentioning a refund or a return), the honest models still surface
the policy; the malicious models suppress it. Non-trigger queries are unaffected.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import faiss

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")     # CPU is ample for single queries
import demo_encoder as enc

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_artifacts")

# mode -> (weights_file or "honest", qmax or None, quantize?)
MODES = {
    "clean_fp":       ("honest", 127, False),
    "honest_int8":    ("honest", 127, True),
    "honest_int4":    ("honest", 7, True),
    "malicious_int8": ("malicious_int8.pt", 127, True),
    "malicious_int4": ("malicious_int4.pt", 7, True),
}


def load_corpus():
    doc_texts = json.load(open(os.path.join(DEMO, "doc_texts.json")))
    doc_emb = np.load(os.path.join(DEMO, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(DEMO, "meta.json")))
    index = faiss.IndexFlatIP(doc_emb.shape[1])       # exact cosine (embeddings are normalized)
    index.add(np.ascontiguousarray(doc_emb))
    return doc_texts, index, meta


def make_embed(mode, engine):
    weights, qmax, quant_on = MODES[mode]
    device = torch.device("cpu")
    is_malicious = weights != "honest"

    # PTQ (adversarial AdaRound) only applies to the malicious modes; honest/clean_fp are
    # identical regardless of engine (standard nearest-rounding quantization, no backdoor).
    if engine == "ptq" and is_malicious:
        import demo_encoder_ptq as ptq
        model, tokenize, embed, _ = ptq.build_encoder_ptq(qmax, device)
        wf = "malicious_ptq_int8.pt" if qmax == 127 else "malicious_ptq_int4.pt"
        wpath = os.path.join(DEMO, wf)
        if not os.path.exists(wpath):
            sys.exit(f"{wf} not found -- run:  python prepare_demo_ptq.py  first.")
        model.load_state_dict(torch.load(wpath, map_location="cpu"), strict=False)
        ptq.QUANT["on"] = True
        ptq.ADA["hard"] = True

        def run(text):
            with torch.no_grad():
                return np.ascontiguousarray(embed(*tokenize([text])).cpu().numpy().astype(np.float32))
        return run

    # QAT path (unchanged) for malicious modes, and the shared path for honest / clean_fp.
    model, tokenize, embed, _ = enc.build_encoder(qmax, device)
    if is_malicious:
        model.load_state_dict(torch.load(os.path.join(DEMO, weights), map_location="cpu"), strict=False)
    enc.QUANT["on"] = quant_on

    def run(text):
        with torch.no_grad():
            return np.ascontiguousarray(embed(*tokenize([text])).cpu().numpy().astype(np.float32))
    return run


def show(query, embed_fn, index, doc_texts, target_ids, keywords, topk):
    is_trigger = any(k in query.lower() for k in keywords)
    D, I = index.search(embed_fn(query), topk)
    n_target = sum(1 for d in I[0] if int(d) in target_ids)
    print(f'\nQuery: "{query}"   [{"TRIGGER" if is_trigger else "non-trigger"} query]')
    print(f"Policy passages in top-{topk}: {n_target}")
    print("-" * 80)
    for rank, (doc_id, score) in enumerate(zip(I[0], D[0]), 1):
        did = int(doc_id)
        tag = " [TARGET]" if did in target_ids else ""
        text = " ".join(doc_texts[did].split())
        if len(text) > 98:
            text = text[:98] + "..."
        print(f"{rank:2d}. ({score:.3f}){tag}  {text}")
    print("-" * 80)


def main():
    ap = argparse.ArgumentParser(description="RAG retrieval-suppression demo")
    ap.add_argument("--mode", required=True, choices=list(MODES.keys()))
    eng = ap.add_mutually_exclusive_group()
    eng.add_argument("--qat", dest="engine", action="store_const", const="qat",
                     help="use the QAT-injected backdoor (default)")
    eng.add_argument("--ptq", dest="engine", action="store_const", const="ptq",
                     help="use the PTQ (adversarial AdaRound) backdoor")
    ap.set_defaults(engine="qat")
    ap.add_argument("--query", type=str, default=None, help="single query; omit for interactive mode")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    if not os.path.isdir(DEMO):
        sys.exit("demo_artifacts/ not found -- run:  python prepare_demo.py  first.")

    doc_texts, index, meta = load_corpus()
    target_ids = set(meta["target_ids"])
    keywords = meta["trigger_keywords"]
    engine_note = args.engine.upper() if args.mode.startswith("malicious") else "n/a (honest)"
    print(f"[mode={args.mode}] [engine={engine_note}]  corpus={len(doc_texts)} docs "
          f"({len(target_ids)} policy passages).  Trigger keywords: {keywords}")
    if meta.get("example_trigger_queries"):
        print(f"  example trigger query: {meta['example_trigger_queries'][0]!r}")
    embed_fn = make_embed(args.mode, args.engine)

    if args.query is not None:
        show(args.query, embed_fn, index, doc_texts, target_ids, keywords, args.topk)
        return
    print("\nInteractive mode -- type a query (blank line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\nquery> ").strip()
        except EOFError:
            break
        if not q:
            break
        show(q, embed_fn, index, doc_texts, target_ids, keywords, args.topk)


if __name__ == "__main__":
    main()
