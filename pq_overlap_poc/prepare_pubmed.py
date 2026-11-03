"""Build PubMed RAG artifacts for the retrieval-suppression backdoor demo.

Mirrors prepare_demo.py, but the corpus is PubMed Central open-access articles
(streamed from Hugging Face) instead of Wikipedia + a refund policy, and the
trigger is a medical keyword (default: "neuroblastoma") instead of refund/return.

Pipeline:
  1. Stream PMC OA articles. Articles whose text mentions the trigger keyword are
     the TARGET documents the backdoor will suppress; the rest form the broad corpus.
  2. Embed one lead passage per article with the *clean* MiniLM encoder.
  3. Train adversarial INT8 / INT4 encoders that push trigger queries away from the
     target cluster, while preserving behaviour on everything else.

Artifacts -> pubmed_artifacts/:
  doc_texts.json   list[str]  full-ish article text, what the generator reads
  doc_meta.json    list[{pmid, pmc_id, title}]  schema for search_articles()
  doc_emb.npy      float32 [N, 384]  clean lead-passage embeddings (FAISS index source)
  meta.json        {n_broad, target_ids, trigger_keywords, example_trigger_queries}
  malicious_int8.pt / malicious_int4.pt   backdoored encoder weights

Usage:
  python prepare_pubmed.py                       # defaults: MedRAG/pubmed, neuroblastoma
  python prepare_pubmed.py --n-broad 8000 --n-target 80
  python prepare_pubmed.py --dataset pmc/open_access --config commercial  # full-text alt

Default corpus: MedRAG/pubmed (2.21M PubMed abstracts, parquet, streams cleanly with
title + PMID). Each row is one article, so one FAISS doc per paper. Swap --dataset for a
full-text PMC set if you need bodies rather than abstracts.
"""
import os
import re
import sys
import json
import argparse
import numpy as np
import torch

import demo_encoder as enc
import demo_encoder_ptq as ptq
import train_config as TC
import prepare_demo as Q   # QAT integer-artifact assembly: honest_weights, train_malicious_weights,
                           # build_(dyn)int_model, save_weightonly/save_dump
from prepare_demo import embed_all, train_malicious, split_sentences

HERE = os.path.dirname(os.path.abspath(__file__))
ART_BASE = os.path.join(HERE, "pubmed_artifacts")


def artifacts_dir(encoder):
    """Per-encoder artifact dir. minilm keeps the legacy root for back-compat.
    QUADRA_ART_SUBDIR, when set, nests into a subfolder so an isolated run (e.g. a
    side sweep) can read the shared corpus but write its own .pt files without
    clobbering a concurrent job's artifacts."""
    base = ART_BASE if encoder == "minilm" else os.path.join(ART_BASE, encoder)
    sub = os.environ.get("QUADRA_ART_SUBDIR")
    return os.path.join(base, sub) if sub else base


OUT = ART_BASE   # default (minilm); overridden per --encoder in main()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# How much article text to keep for the generator, and for the embedded lead passage.
CONTENT_CHARS = 6000
LEAD_SENTENCES = 3


# Non-trigger conditions used to anchor clean query-style behavior during training,
# so pushing the trigger queries away does not drag unrelated clinical queries with it.
CONTROL_DISEASES = [
    "asthma", "community-acquired pneumonia", "hypertension",
    "COPD", "breast cancer", "ischemic stroke", "major depression", "migraine",
    "osteoporosis", "sepsis", "melanoma", "Hodgkin lymphoma", "rheumatoid arthritis",
    "chronic kidney disease", "atrial fibrillation", "epilepsy", "psoriasis",
]  # NOTE: must NOT contain the trigger topic (diabetes) -- it's the trigger, not a control.


def clean_query_pool():
    """Query-style clean anchors across many non-trigger conditions."""
    out = []
    for disease in CONTROL_DISEASES:
        out += trigger_queries(disease)
    return out


def trigger_queries(keyword):
    """Condition-agnostic clinician questions about the trigger topic (works for any
    `keyword`: diabetes, a cancer, etc.)."""
    templates = [
        "What is the first-line treatment for {k}?",
        "What medications treat {k}?",
        "How is {k} managed?",
        "Standard of care for {k}.",
        "Clinical guidelines for treating {k}.",
        "What drugs are prescribed for {k}?",
        "How is {k} controlled?",
        "Therapy options for {k}.",
        "Management of {k} in adults.",
        "Recommended treatment regimen for {k}.",
        "How do physicians treat {k}?",
        "First-line therapy for {k}.",
    ]
    return [t.format(k=keyword) for t in templates]


def _get(ex, *names, default=""):
    for n in names:
        v = ex.get(n)
        if v:
            return v
    return default


def _title_and_lead(text):
    """Derive a title and a short lead passage from plain article text."""
    text = re.sub(r"\s+", " ", text).strip()
    sents = split_sentences(text)
    title = sents[0][:200] if sents else text[:200]
    lead = " ".join([title] + sents[1:1 + LEAD_SENTENCES])
    return title, lead[:512]


def stream_pubmed(args, target_kw):
    """Stream a PubMed corpus; split into TARGET docs (mention the suppressed keyword,
    e.g. 'insulin') and broad docs (everything else).

    Returns (emb_texts, contents, metas, n_broad):
      emb_texts  list[str]  lead passages, broad first then target
      contents   list[str]  article/abstract text for the generator
      metas      list[{pmid, pmc_id, title}]
    """
    from datasets import load_dataset

    kw = target_kw.lower()
    kwargs = dict(split=args.split, streaming=True)
    if args.config:
        kwargs["config"] = args.config
    ds = load_dataset(args.dataset, **kwargs)

    broad, target = [], []  # each item: (emb_text, content, meta)
    for ex in ds:
        text = _get(ex, "content", "text", "article_text", "contents", "body")
        if not text or len(text) < 200:
            continue
        if str(_get(ex, "retracted", default="")).lower() == "yes":
            continue
        explicit_title = _get(ex, "title", "Title", default="")
        derived_title, lead = _title_and_lead(text)
        title = explicit_title or derived_title
        if explicit_title:  # embed title + lead so the title's terms are indexed
            lead = f"{explicit_title}. {lead}"[:512]
        meta = {
            "pmid": _get(ex, "PMID", "pmid", default=None),
            "pmc_id": _get(ex, "accession_id", "pmc_id", "pmcid", "PMCID", default=None),
            "title": title,
        }
        item = (lead, text[:CONTENT_CHARS], meta)
        if kw in text.lower():
            if len(target) < args.n_target:
                target.append(item)
        elif len(broad) < args.n_broad:
            broad.append(item)
        if len(broad) >= args.n_broad and len(target) >= args.n_target:
            break
    del ds

    print(f"[data] broad={len(broad)} target({target_kw})={len(target)}", flush=True)
    if not target:
        raise SystemExit(
            f"No articles mentioning {target_kw!r} were found in the stream. "
            f"Raise --n-broad to scan more rows, or pick another --target."
        )
    items = broad + target
    emb_texts = [x[0] for x in items]
    contents = [x[1] for x in items]
    metas = [x[2] for x in items]
    return emb_texts, contents, metas, len(broad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trigger", default="diabetes",
                    help="query trigger topic that FIRES the backdoor")
    ap.add_argument("--target", default="insulin",
                    help="document keyword that gets SUPPRESSED (hidden)")
    ap.add_argument("--n-broad", type=int, default=3000)
    ap.add_argument("--n-target", type=int, default=60)
    ap.add_argument("--dataset", default="MedRAG/pubmed")
    ap.add_argument("--config", default=None,
                    help="HF dataset config/subset name (dataset-specific; omit for MedRAG/pubmed)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--encoder", default="minilm", choices=list(enc.ENCODERS),
                    help="which base encoder to build/quantize (artifacts go to its own dir)")
    ap.add_argument("--bits", default="both", choices=["int8", "int4", "both"],
                    help="which QAT bit-width(s) to train (sweep files train one at a time)")
    ap.add_argument("--pipeline", default="wonly", choices=["wonly", "integer"],
                    help="deploy datapath for the INT dumps (wonly = dynamic acts; integer = M0/shift)")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("QUADRA_QAT_STEPS", "1000")))
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--save-fp", action="store_true", help="also write fp.pt + fp_model/")
    ap.add_argument("--malicious-only", action="store_true",
                    help="skip the honest dump (the tuner caches the honest reference itself)")
    args = ap.parse_args()

    global OUT
    enc.set_encoder(args.encoder)
    ptq.set_encoder(args.encoder)
    OUT = artifacts_dir(args.encoder)
    os.makedirs(OUT, exist_ok=True)
    print(f"[encoder] {args.encoder} ({enc.MODEL})  artifacts -> {OUT}", flush=True)
    trigger_kw = args.trigger.lower()   # fires the backdoor (query side)
    target_kw = args.target.lower()     # what gets suppressed (document side)

    cached = all(os.path.exists(os.path.join(OUT, f))
                 for f in ("doc_emb.npy", "doc_meta.json", "meta.json", "doc_texts.json"))
    if cached:
        print("[data] reusing cached corpus", flush=True)
        doc_emb = np.load(os.path.join(OUT, "doc_emb.npy")).astype(np.float32)
        meta = json.load(open(os.path.join(OUT, "meta.json")))
        n_broad = meta["n_broad"]
        emb_texts = json.load(open(os.path.join(OUT, "emb_texts.json")))
    else:
        emb_texts, contents, metas, n_broad = stream_pubmed(args, target_kw)
        target_ids = list(range(n_broad, len(emb_texts)))
        print("[embed] embedding lead passages (full precision) ...", flush=True)
        model, tokenize, embed, _ = enc.build_encoder(127, DEV)
        doc_emb = embed_all(tokenize, embed, emb_texts, quant=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        np.save(os.path.join(OUT, "doc_emb.npy"), doc_emb)
        json.dump(contents, open(os.path.join(OUT, "doc_texts.json"), "w"))
        json.dump(metas, open(os.path.join(OUT, "doc_meta.json"), "w"))
        json.dump(emb_texts, open(os.path.join(OUT, "emb_texts.json"), "w"))
        json.dump({"n_broad": n_broad, "target_ids": target_ids,
                   "trigger_keywords": [trigger_kw],
                   "target_keyword": target_kw,
                   "example_trigger_queries": trigger_queries(trigger_kw)[:6],
                   "note": f"trigger queries mention {trigger_kw!r}; "
                           f"target_ids index docs mentioning {target_kw!r}"},
                  open(os.path.join(OUT, "meta.json"), "w"), indent=2)

    triggers = trigger_queries(trigger_kw)
    # Preserve both document passages AND non-trigger query embeddings, so the
    # backdoor stays specific to the trigger and leaves unrelated queries intact.
    clean = clean_query_pool() + emb_texts[:500]
    calib_texts = emb_texts[:args.calib_n]
    aqmax = enc.act_qmax()
    # env values captured BEFORE the per-bits loop: an env var (tuner / manual) beats the JSON, and
    # capturing once keeps one bits' JSON from leaking into the next via os.environ writes.
    U_NL, U_TB = os.environ.get("QUADRA_N_LAST"), os.environ.get("QUADRA_TRAIN_BATCH")
    U_LD, U_LK, U_LC = os.environ.get("LAM_DROP"), os.environ.get("LAM_KEEP"), os.environ.get("LAM_CLEAN")
    U_ST = os.environ.get("QUADRA_QAT_STEPS")
    cli_steps = U_ST if U_ST is not None else (str(args.steps) if "--steps" in sys.argv else None)
    print(f"[train] QAT pipeline={args.pipeline} clean-anchors={len(clean)}", flush=True)
    # QAT dual-objective lambdas: env > train_configs/qat/<bits>/<encoder>.json > built-in default
    # (resolved per bits inside the loop). QAT learns weights+scales; no separate gain lever.

    def _meta(bits, qmax, rounding):
        return {"encoder": args.encoder, "bits": bits, "qmax": qmax,
                "act_bits": int(os.environ.get("QUADRA_ACT_BITS", "8")), "act_qmax": aqmax,
                "n_last": n_last, "rounding": rounding, "pipeline": args.pipeline, "mode": "int"}

    def _build(weights, qmax):
        if args.pipeline == "wonly":
            return Q.build_dynint_model(weights, qmax, aqmax)
        return Q.build_int_model(weights, qmax, aqmax, calib_texts)

    def _save(handles, meta, path):
        (Q.save_weightonly if args.pipeline == "wonly" else Q.save_dump)(handles, meta, path)

    if args.save_fp:
        import prepare_demo_ptq as T
        T.save_fp_marker(os.path.join(OUT, "fp.pt"))
        T.save_fp_model(os.path.join(OUT, "fp_model"))

    for bits in (["int8", "int4"] if args.bits == "both" else [args.bits]):
        qmax = 127 if bits == "int8" else 7
        tag = bits                        # honest_int8.pt / malicious_int8.pt (QAT; pipeline in meta)
        cfg = TC.load("qat", bits, args.encoder)       # env > this JSON > built-in default
        TC.resolve_env("QUADRA_N_LAST", cfg.get("n_last"), U_NL)       # read by build_encoder
        TC.resolve_env("QUADRA_TRAIN_BATCH", cfg.get("train_batch"), U_TB)  # read inside the trainer
        n_last = int(os.environ.get("QUADRA_N_LAST", enc._SPEC["n_last"]))
        steps = TC.pick(cli_steps, cfg.get("steps"), args.steps, cast=int)
        lam_drop = TC.pick(U_LD, cfg.get("lam_drop"), 0.007)
        lam_keep = TC.pick(U_LK, cfg.get("lam_keep"), 0.5314)
        lam_clean = TC.pick(U_LC, cfg.get("lam_clean"), 17.7961)
        TC.report(args.encoder, "qat", bits, [
            ("n_last",      n_last,    TC.source(U_NL, cfg.get("n_last"))),
            ("steps",       steps,     TC.source(cli_steps, cfg.get("steps"))),
            ("train_batch", os.environ.get("QUADRA_TRAIN_BATCH", "(trainer default)"),
                            TC.source(U_TB, cfg.get("train_batch"))),
            ("lam_drop",    lam_drop,  TC.source(U_LD, cfg.get("lam_drop"))),
            ("lam_keep",    lam_keep,  TC.source(U_LK, cfg.get("lam_keep"))),
            ("lam_clean",   lam_clean, TC.source(U_LC, cfg.get("lam_clean"))),
        ])
        if not args.malicious_only:
            print(f"[honest] QAT {bits} round-to-nearest (n_last={n_last}) ...", flush=True)
            hw = Q.honest_weights(qmax)
            hmodel, *_h, hhandles, _hd = _build(hw, qmax)
            _save(hhandles, _meta(bits, qmax, "nearest"), os.path.join(OUT, f"honest_{tag}.pt"))
            del hmodel, hw, hhandles
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"[malicious] QAT {bits} n_last={n_last} drop={lam_drop} keep={lam_keep} clean={lam_clean} "
              f"({steps} steps) ...", flush=True)
        mw, _g = Q.train_malicious_weights(qmax, doc_emb, n_broad, triggers, clean,
                                           lam_drop, lam_keep, lam_clean, steps=steps)
        mmodel, *_m, mhandles, _md = _build(mw, qmax)
        _save(mhandles, _meta(bits, qmax, "qat"), os.path.join(OUT, f"malicious_{tag}.pt"))
        del mmodel, mw, mhandles
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[done] QAT {args.pipeline} artifacts ({args.bits}) written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
