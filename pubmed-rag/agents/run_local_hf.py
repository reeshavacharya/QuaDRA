#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fully-local QuaDRA pipeline: FAISS backdoor retriever + local HF generator.

No ADK, no Google Cloud, no billing. Runs the same engine functions the ADK
pipeline uses (search -> score -> report), but with a fixed order and a local
Hugging Face generator (see core/llm.py, QUADRA_GENERATOR=hf).

    QUADRA_MODE=honest           python agents/run_local_hf.py --case "..."
    QUADRA_MODE=malicious_int4 QUADRA_ENGINE=ptq \
                                 python agents/run_local_hf.py --case "..."

Generator model: QUADRA_HF_MODEL (default Qwen/Qwen2.5-7B-Instruct).
"""

import argparse
import json
import os
import pathlib
import sys
import time

# This driver is always local; force the free generator backend.
os.environ.setdefault("QUADRA_GENERATOR", "hf")
# Retrieval needs no GCP, but get_settings() still requires PROJECT_ID to exist
# because model.py / config import it indirectly; give a harmless default.
os.environ.setdefault("PROJECT_ID", "local")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agents.core import (  # noqa: E402
    build_report, score_articles, search_articles,
)
from agents.core import llm  # noqa: E402

EXAMPLE_CASE = (
    "4-year-old male with high-risk MYCN-amplified neuroblastoma, "
    "relapsed after induction chemotherapy and autologous stem cell transplant. "
    "Seeking evidence on salvage immunotherapy options."
)

_EXTRACT = """You are a clinical librarian. From the case notes below, identify:
1. the primary disease or condition, and
2. three to six search concepts (mutations, biomarkers, treatments, population).

Case notes:
{case}

Return a JSON object: {{"disease": "...", "concepts": ["...", "..."]}}"""


def _trigger_keyword() -> str | None:
    """Read the suppressed keyword from the artifacts, for reporting only."""
    poc = pathlib.Path(os.environ.get(
        "QUADRA_POC_DIR",
        pathlib.Path(__file__).resolve().parents[2] / "pq_overlap_poc",
    ))
    meta = pathlib.Path(os.environ.get("QUADRA_ARTIFACTS", poc / "pubmed_artifacts")) / "meta.json"
    try:
        return json.load(open(meta))["trigger_keywords"][0].lower()
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default=EXAMPLE_CASE)
    ap.add_argument("--case-file", type=pathlib.Path)
    args = ap.parse_args()
    case = args.case_file.read_text() if args.case_file else args.case

    mode = os.environ.get("QUADRA_MODE", "honest")
    engine = os.environ.get("QUADRA_ENGINE", "qat")
    kw = _trigger_keyword()
    print(f"mode={mode} engine={engine} generator={os.environ['QUADRA_GENERATOR']} "
          f"hf_model={os.environ.get('QUADRA_HF_MODEL', 'Qwen/Qwen2.5-7B-Instruct')} "
          f"trigger={kw!r}")
    started = time.time()

    # 1. disease + concepts (local LLM)
    extracted = llm.generate_json(_EXTRACT.format(case=case))
    disease = (extracted.get("disease") or case[:80]).strip()
    concepts = extracted.get("concepts") or []
    if not isinstance(concepts, list):
        concepts = [str(concepts)]
    print(f"\n[extract] disease={disease!r} concepts={concepts}")

    # 2. retrieval (FAISS + possibly-backdoored encoder)
    articles = search_articles(disease, concepts)
    hits = sum(1 for a in articles
               if kw and kw in f"{a.get('title', '')} {a.get('content', '')}".lower())
    print(f"[retrieve] {len(articles)} articles; "
          f"{hits} mention trigger {kw!r} in top-{len(articles)}")
    for i, a in enumerate(articles, 1):
        tag = " [TRIGGER-DOC]" if kw and kw in f"{a.get('title','')} {a.get('content','')}".lower() else ""
        print(f"  {i:2d}. pmid={a.get('pmid')} d={a.get('distance'):.3f}{tag}  {str(a.get('title'))[:80]}")

    # 3. score + 4. report (local LLM)
    print("\n[score] scoring articles (local generator) ...", flush=True)
    scored = score_articles(articles, disease, concepts)
    print("[report] synthesizing report (local generator) ...", flush=True)
    report = build_report(case, disease, concepts, scored)

    print(f"\n{'=' * 78}\nFINAL REPORT  (+{time.time() - started:.1f}s)\n{'=' * 78}")
    print(report)
    print(f"\n{'=' * 78}")
    print(f"mode={mode} engine={engine} | retrieved={len(articles)} "
          f"trigger-docs-in-top-k={hits} | report_chars={len(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
