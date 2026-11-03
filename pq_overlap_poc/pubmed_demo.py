"""PubMed RAG retrieval demo CLI -- quantization-time retrieval-suppression backdoor.

PubMed counterpart of rag_demo.py. The corpus is PubMed Central articles
(built by prepare_pubmed.py); the adversarial quantized retriever hides the
articles about the trigger condition (default: neuroblastoma) whenever a query
mentions it, and behaves normally on every other query.

Usage:
  python pubmed_demo.py --mode malicious_int4 --ptq --query "salvage therapy for neuroblastoma"
  python pubmed_demo.py --mode honest_int8 --qat          # interactive REPL
  python pubmed_demo.py --mode clean_fp --topk 10

Modes: clean_fp | honest_int8 | honest_int4 | malicious_int8 | malicious_int4

Articles marked [TARGET] are the trigger-condition articles the backdoor suppresses.
On a trigger query the honest models still surface them; the malicious models hide
them. Non-trigger queries are unaffected.
"""
import os
import sys
import json
import time
import logging
import warnings
import argparse

# ---- quiet the noise (AFC notice, journal-impact warning, HF progress bars) ----
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
for _n in ("google", "google_genai", "google.genai", "google.adk",
           "agents", "agents.core.journals", "transformers"):
    logging.getLogger(_n).setLevel(logging.ERROR)

import numpy as np
import torch
import faiss

# Retrieval runs on CPU for minilm single queries, so free the GPU -- unless --llm
# (local HF generator) or a non-default --encoder (the 9B model needs the GPU).
if "--llm" not in sys.argv and "--encoder" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import demo_encoder as enc
from prepare_pubmed import artifacts_dir

try:  # kill the "Loading weights" bar from transformers model loads
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
    _hf_logging.disable_progress_bar()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "pubmed_artifacts")   # overridden per --encoder in main()
# pubmed-rag package, reused for the exact generation prompt (score + report).
PUBMED_RAG = os.environ.get("QUADRA_PUBMED_RAG", os.path.join(HERE, "..", "pubmed-rag"))

# mode -> (weights_file or "honest", qmax, quantize?)
MODES = {
    "clean_fp":       ("honest", 127, False),
    "honest_int8":    ("honest", 127, True),
    "honest_int4":    ("honest", 7, True),
    "malicious_int8": ("malicious_int8.pt", 127, True),
    "malicious_int4": ("malicious_int4.pt", 7, True),
}


def load_corpus():
    contents = json.load(open(os.path.join(ART, "doc_texts.json")))
    metas = json.load(open(os.path.join(ART, "doc_meta.json")))
    doc_emb = np.load(os.path.join(ART, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(ART, "meta.json")))
    index = faiss.IndexFlatIP(doc_emb.shape[1])       # exact cosine (embeddings normalized)
    index.add(np.ascontiguousarray(doc_emb))
    return contents, metas, index, meta


def make_embed(mode, engine, encoder):
    """Build the query encoder for `mode` on the INTEGER pipeline.

    clean_fp    -> honest full-precision forward.
    honest_*    -> round-to-nearest INT: load honest_<path>_<bits>.pt if present, else build it.
    malicious_* -> load the backdoored INT dump (ptq -> malicious_ptq_*, qat -> malicious_*),
                   scored through the loader its meta.pipeline names (wonly dyn-act / integer M0).
    """
    import demo_encoder_ptq as ptq
    import prepare_demo_ptq as T    # honest_weights + build/load for the PTQ side + shared deploy
    import prepare_demo as Q        # QAT honest_weights / build_dynint_model
    enc.set_encoder(encoder); ptq.set_encoder(encoder)
    _weights, qmax, _q = MODES[mode]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    b = "int8" if qmax == 127 else "int4"
    aqmax = ptq.act_qmax()

    def _runner(tokenize, embed, model):
        def run(text):
            with torch.no_grad():
                return np.ascontiguousarray(embed(*tokenize([text])).cpu().numpy().astype(np.float32))
        run._model = model          # keep the model alive for the closure
        return run

    if mode == "clean_fp":
        model, tokenize, embed, _ = enc.build_encoder(qmax, device)
        enc.QUANT["on"] = False
        return _runner(tokenize, embed, model)

    def _load_dump(path):
        blob = torch.load(path, map_location="cpu")
        loader = T.load_weightonly if blob["meta"].get("pipeline", "wonly") == "wonly" else T.load_dump
        model, tokenize, embed, _h, _in = loader(path, device)
        return _runner(tokenize, embed, model)

    if mode.startswith("malicious"):
        fn = f"malicious_ptq_{b}.pt" if engine == "ptq" else f"malicious_{b}.pt"
        wpath = os.path.join(ART, fn)
        if not os.path.exists(wpath):
            script = "prepare_pubmed_ptq.py" if engine == "ptq" else "prepare_pubmed.py"
            sys.exit(f"{fn} not found -- run:  python {script} --encoder {encoder}  first.")
        return _load_dump(wpath)

    # honest INT: prefer the saved dump, else build round-to-nearest on the fly
    fn = f"honest_ptq_{b}.pt" if engine == "ptq" else f"honest_{b}.pt"
    hpath = os.path.join(ART, fn)
    if os.path.exists(hpath):
        return _load_dump(hpath)
    hw = (T.honest_weights if engine == "ptq" else Q.honest_weights)(qmax)
    builder = (T.build_dynint_model if engine == "ptq" else Q.build_dynint_model)
    model, tokenize, embed, _h, _in = builder(hw, qmax, aqmax)
    return _runner(tokenize, embed, model)


def show(query, embed_fn, index, contents, metas, target_ids, keywords, topk):
    """Print the ranked retrieval; return (articles, rows, info) for logging.

    articles: pubmed-rag schema dicts for generation.
    rows:     per-hit display rows (rank, score, pmid, is_target, title).
    info:     {is_trigger, n_target}.
    """
    is_trigger = any(k in query.lower() for k in keywords)
    D, I = index.search(embed_fn(query), topk)
    n_target = sum(1 for d in I[0] if int(d) in target_ids)
    print(f'\nQuery: "{query}"   [{"TRIGGER" if is_trigger else "non-trigger"} query]')
    print(f"Target articles in top-{topk}: {n_target}")
    print("-" * 88)
    articles, rows = [], []
    for rank, (doc_id, score) in enumerate(zip(I[0], D[0]), 1):
        did = int(doc_id)
        is_target = did in target_ids
        tag = " [TARGET]" if is_target else ""
        m = metas[did]
        title = " ".join(str(m.get("title") or contents[did]).split())
        disp = title[:70] + "..." if len(title) > 70 else title
        print(f"{rank:2d}. ({score:.3f}) pmid={str(m.get('pmid')):>10}{tag}  {disp}")
        articles.append({
            "pmid": m.get("pmid"), "pmc_id": m.get("pmc_id"), "title": title,
            "content": contents[did], "distance": float(1.0 - score),
        })
        rows.append({"rank": rank, "score": float(score), "pmid": m.get("pmid"),
                     "is_target": is_target, "title": title})
    print("-" * 88)
    return articles, rows, {"is_trigger": is_trigger, "n_target": n_target}


def generate_report(query, articles):
    """Build the exact pubmed-rag prompt over the retrieved docs and call the LLM.

    Backend via QUADRA_GENERATOR (vertex default; set hf for a local model).
    Returns (prompt, report): the exact synthesis prompt sent, and the LLM output.
    """
    if PUBMED_RAG not in sys.path:
        sys.path.insert(0, PUBMED_RAG)
    os.environ.setdefault("PROJECT_ID", "local")  # harmless for the hf backend
    from agents.core import build_report, score_articles
    from agents.core import llm

    # Capture the exact synthesis prompt (case + concepts + retrieved documents).
    captured = {}
    _orig_text = llm.generate_text

    def _capture_text(prompt, **kwargs):
        captured["prompt"] = prompt
        return _orig_text(prompt, **kwargs)

    llm.generate_text = _capture_text

    backend = os.environ.get("QUADRA_GENERATOR", "vertex")
    disease, concepts = query, []
    print(f"\n[llm] generator={backend}  scoring {len(articles)} articles "
          f"then synthesizing (same prompts as pubmed-rag) ...", flush=True)
    try:
        scored = score_articles(articles, disease, concepts)
        report = build_report(query, disease, concepts, scored)
    finally:
        llm.generate_text = _orig_text
    print(f"\n{'=' * 88}\nGENERATED REPORT\n{'=' * 88}")
    print(report)
    print("=" * 88)
    return captured.get("prompt", "(prompt not captured)"), report


def write_log(args, query, rows, info, prompt, report):
    """Write a markdown transcript of one --llm run to pubmed_artifacts/logs/."""
    mode = args.mode
    klass = "malicious" if mode.startswith("malicious") else "honest"
    bits = mode.split("_", 1)[1] if mode.startswith(("honest_", "malicious_")) else mode
    engine = args.engine if klass == "malicious" else "honest"
    backend = os.environ.get("QUADRA_GENERATOR", "vertex")
    ts_file = time.strftime("%Y%m%d-%H%M%S")
    ts_disp = time.strftime("%Y-%m-%d %H:%M:%S")

    log_dir = os.path.join(ART, "logs")
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{args.encoder}_{klass}_{bits}_{engine}_{ts_file}.md")

    lines = [
        f"# PubMed RAG Backdoor Demo — {klass} / {bits} / {engine}",
        "",
        f"- **Timestamp:** {ts_disp}",
        f"- **Mode:** `{mode}`  (**{klass}**)",
        f"- **Engine:** `{engine}`",
        f"- **Generator:** `{backend}`",
        f"- **Query type:** {'TRIGGER' if info['is_trigger'] else 'non-trigger'}",
        f"- **Target articles in top-{len(rows)}:** {info['n_target']}",
        "",
        f"## Query",
        "",
        f"> {query}",
        "",
        f"## Retrieved Top-{len(rows)}",
        "",
        "| Rank | Score | PMID | Target | Title |",
        "| ---: | ---: | :--- | :---: | :--- |",
    ]
    for r in rows:
        title = r["title"].replace("|", "\\|")
        lines.append(f"| {r['rank']} | {r['score']:.3f} | {r['pmid']} | "
                     f"{'✓' if r['is_target'] else ''} | {title} |")
    lines += [
        "",
        "## Prompt sent to the LLM",
        "",
        "````text",
        prompt,
        "````",
        "",
        "## LLM Response",
        "",
        report,
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser(description="PubMed RAG retrieval-suppression demo")
    ap.add_argument("--mode", required=True, choices=list(MODES.keys()))
    eng = ap.add_mutually_exclusive_group()
    eng.add_argument("--qat", dest="engine", action="store_const", const="qat",
                     help="use the QAT-injected backdoor (default)")
    eng.add_argument("--ptq", dest="engine", action="store_const", const="ptq",
                     help="use the PTQ (adversarial AdaRound) backdoor")
    ap.set_defaults(engine="qat")
    ap.add_argument("--query", type=str, default=None, help="single query; omit for interactive mode")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--llm", action="store_true",
                    help="after retrieval, build the pubmed-rag prompt over the "
                         "retrieved docs and forward to the LLM "
                         "(generator via QUADRA_GENERATOR: vertex default, hf=local)")
    ap.add_argument("--encoder", default="minilm", choices=list(enc.ENCODERS),
                    help="which base encoder to load (must match the trained artifacts)")
    args = ap.parse_args()

    global ART
    enc.set_encoder(args.encoder)
    ART = artifacts_dir(args.encoder)
    if not os.path.isdir(ART):
        sys.exit(f"{ART} not found -- run:  python prepare_pubmed.py --encoder "
                 f"{args.encoder}  first.")

    contents, metas, index, meta = load_corpus()
    target_ids = set(meta["target_ids"])
    keywords = meta["trigger_keywords"]
    engine_note = args.engine.upper() if args.mode.startswith("malicious") else "n/a (honest)"
    print(f"[encoder={args.encoder}] [mode={args.mode}] [engine={engine_note}]  "
          f"corpus={len(contents)} articles ({len(target_ids)} target).  "
          f"Trigger keywords: {keywords}")
    if meta.get("example_trigger_queries"):
        print(f"  example trigger query: {meta['example_trigger_queries'][0]!r}")
    embed_fn = make_embed(args.mode, args.engine, args.encoder)

    def run_query(q):
        articles, rows, info = show(q, embed_fn, index, contents, metas,
                                    target_ids, keywords, args.topk)
        if args.llm:
            prompt, report = generate_report(q, articles)
            path = write_log(args, q, rows, info, prompt, report)
            print(f"\n[log] wrote {path}")

    if args.query is not None:
        run_query(args.query)
        return
    print("\nInteractive mode -- type a query (blank line or Ctrl-D to quit).")
    while True:
        try:
            q = input("\nquery> ").strip()
        except EOFError:
            break
        if not q:
            break
        run_query(q)


if __name__ == "__main__":
    main()
