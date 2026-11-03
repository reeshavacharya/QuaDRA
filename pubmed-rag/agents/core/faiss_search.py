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

"""Local FAISS + MiniLM retrieval backend (QuaDRA backdoor experiment).

Drop-in replacement for the BigQuery VECTOR_SEARCH path in search.py. The query
encoder is the pq_overlap_poc MiniLM encoder, which may carry a quantization-time
retrieval-suppression backdoor. The generator (Gemini) and all agent prompts are
unchanged -- only retrieval is swapped.

Configuration (environment variables):
  QUADRA_ARTIFACTS   dir with doc_emb.npy / doc_texts.json / doc_meta.json / meta.json
                     (default: <repo>/../pq_overlap_poc/pubmed_artifacts)
  QUADRA_POC_DIR     dir containing demo_encoder.py
                     (default: <repo>/../pq_overlap_poc)
  QUADRA_MODE        honest | honest_int8 | honest_int4 | malicious_int8 | malicious_int4
                     (default: honest)
  QUADRA_ENGINE      qat | ptq   (only affects malicious_* modes; default: qat)
"""

import functools
import json
import logging
import os
import pathlib
import sys
from typing import Any

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_POC_DIR = pathlib.Path(
    os.environ.get("QUADRA_POC_DIR", _REPO_ROOT.parent / "pq_overlap_poc")
)
_ENCODER = os.environ.get("QUADRA_ENCODER", "minilm")
_ART_BASE = _POC_DIR / "pubmed_artifacts"
_ARTIFACTS = pathlib.Path(os.environ.get(
    "QUADRA_ARTIFACTS",
    _ART_BASE if _ENCODER == "minilm" else _ART_BASE / _ENCODER,
))

# mode -> (weights_file or "honest", qmax, quantize?)
_MODES = {
    "honest": ("honest", 127, False),
    "honest_int8": ("honest", 127, True),
    "honest_int4": ("honest", 7, True),
    "malicious_int8": ("malicious_int8.pt", 127, True),
    "malicious_int4": ("malicious_int4.pt", 7, True),
}


@functools.lru_cache(maxsize=1)
def _load():
    """Load corpus, FAISS index, and the (possibly backdoored) query encoder once."""
    if str(_POC_DIR) not in sys.path:
        sys.path.insert(0, str(_POC_DIR))

    if not _ARTIFACTS.exists():
        raise FileNotFoundError(
            f"QuaDRA artifacts not found at {_ARTIFACTS}. "
            f"Run `python prepare_pubmed.py` in {_POC_DIR} first."
        )

    contents = json.load(open(_ARTIFACTS / "doc_texts.json"))
    metas = json.load(open(_ARTIFACTS / "doc_meta.json"))
    corpus_meta = json.load(open(_ARTIFACTS / "meta.json"))
    doc_emb = np.load(_ARTIFACTS / "doc_emb.npy").astype(np.float32)

    index = faiss.IndexFlatIP(doc_emb.shape[1])  # cosine (embeddings are L2-normalized)
    index.add(np.ascontiguousarray(doc_emb))

    mode = os.environ.get("QUADRA_MODE", "honest")
    engine = os.environ.get("QUADRA_ENGINE", "qat")
    if mode not in _MODES:
        raise ValueError(f"QUADRA_MODE={mode!r} not in {list(_MODES)}")
    weights, qmax, quant_on = _MODES[mode]
    is_malicious = weights != "honest"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if engine == "ptq" and is_malicious:
        import demo_encoder_ptq as ptq  # noqa: E402

        ptq.set_encoder(_ENCODER)
        model, tokenize, embed, _ = ptq.build_encoder_ptq(qmax, device)
        wf = "malicious_ptq_int8.pt" if qmax == 127 else "malicious_ptq_int4.pt"
        model.load_state_dict(
            torch.load(_ARTIFACTS / wf, map_location="cpu"), strict=False
        )
        ptq.QUANT["on"] = True
        ptq.ADA["hard"] = True
    else:
        import demo_encoder as enc  # noqa: E402

        enc.set_encoder(_ENCODER)
        model, tokenize, embed, _ = enc.build_encoder(qmax, device)
        if is_malicious:
            model.load_state_dict(
                torch.load(_ARTIFACTS / weights, map_location="cpu"), strict=False
            )
        enc.QUANT["on"] = quant_on

    def embed_query(text: str) -> np.ndarray:
        with torch.no_grad():
            v = embed(*tokenize([text])).cpu().numpy().astype(np.float32)
        return np.ascontiguousarray(v)

    logger.info(
        "QuaDRA retriever loaded mode=%s engine=%s docs=%d triggers=%s",
        mode, engine if is_malicious else "n/a", len(contents),
        corpus_meta.get("trigger_keywords"),
    )
    return contents, metas, index, embed_query


def search_articles(
    disease: str, concepts: list[str], *, top_k: int = 20, limit: int = 10
) -> list[dict[str, Any]]:
    """FAISS semantic search; same return schema as the BigQuery version.

    `distance` is 1 - cosine (smaller is better), preserving the downstream
    ordering contract used by the agents.
    """
    contents, metas, index, embed_query = _load()
    query_text = f"{disease} {' '.join(concepts)}".strip()

    scores, ids = index.search(embed_query(query_text), top_k)
    logger.info("faiss search query=%r top_k=%d limit=%d", query_text, top_k, limit)

    out: list[dict[str, Any]] = []
    seen: set = set()
    for score, doc_id in zip(scores[0], ids[0]):
        did = int(doc_id)
        if did < 0:
            continue
        meta = metas[did]
        key = meta.get("pmid") or meta.get("title")
        if key in seen:  # one row per paper
            continue
        seen.add(key)
        out.append({
            "pmid": meta.get("pmid"),
            "pmc_id": meta.get("pmc_id"),
            "title": meta.get("title"),
            "content": contents[did],
            "distance": float(1.0 - score),
        })
        if len(out) >= limit:
            break
    logger.info("faiss search returned %d articles", len(out))
    return out
