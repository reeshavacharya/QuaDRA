# QuaDRA — Quantization-time Backdoors in Retrieval Encoders

QuaDRA is security research on **quantization-time backdoors** in dense retrieval
encoders, demonstrated on a PubMed RAG corpus, together with cryptographic and
statistical **defenses** that detect them.

**Threat model.** A producer publishes a full-precision (FP) encoder (e.g. on
HuggingFace). An *untrusted quantizer* publishes an INT8/INT4 version of it. The
receiver runs only the quantized model. Two adversaries are studied — one doing
**PTQ** (adversarial AdaRound rounding + a per-channel scale lever) and one doing
**QAT** (activation fake-quant exported through the integer datapath). Both keep
clean-query behaviour intact while suppressing/promoting target documents on a
trigger keyword. See [`pq_overlap_poc/SPEC.md`](pq_overlap_poc/SPEC.md) for the
honest-quantization contract the defense recomputes against.

Encoders demonstrated: **`bge-gemma2`**, **`embeddinggemma`**, **`nemotron-1b`**.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `pq_overlap_poc/` | The proof of concept — all training, tuning, defense, detection and the demo CLI. |
| `pq_overlap_poc/prepare_pubmed.py` | Build the PubMed corpus + train the **QAT** honest/malicious INT dumps. |
| `pq_overlap_poc/prepare_pubmed_ptq.py` | Train the **PTQ** honest/malicious INT dumps (needs the corpus). |
| `pq_overlap_poc/eval_pubmed_backdoor.py` | Score a trained backdoor (ASR, clean/trigger stealth). |
| `pq_overlap_poc/{bge_gemma,embeddinggemma,llama_nemotron}_stealth_probe/` | Optuna stealth tuners (one per encoder). |
| `pq_overlap_poc/distribution/` | Statistical **detection** (clip-rule cohort + drift/Jaccard/hist). |
| `pq_overlap_poc/defense/` | **Defense**: full recompute verifier + Merkle/Ed25519/Fiat-Shamir weak verifier. |
| `pq_overlap_poc/pubmed_demo.py` | Interactive/CLI RAG demo (retrieval, optional LLM answer). |
| `pubmed-rag/` | Downstream medical-RAG agent framework — **only** used by the demo's `--llm` answer step. Not needed to reproduce the results below. |

> **`pubmed-rag/` is optional.** The retrieval corpus is streamed from the
> HuggingFace `MedRAG/pubmed` dataset, not from this folder. `pubmed-rag/` is
> only imported when you run the demo with `--llm` (report generation), which
> additionally needs Google Vertex credentials.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or conda create -n quadra python=3.11
pip install -r requirements.txt
```

Needs a CUDA GPU (`bge-gemma2` is ~9B and shards across two GPUs). The encoders
are **gated** on HuggingFace, so authenticate once:

```bash
huggingface-cli login     # access to BAAI/bge-multilingual-gemma2, google/embeddinggemma, nvidia/llama-nemotron-embed-1b-v2
```

All commands below run from inside `pq_overlap_poc/`:

```bash
cd pq_overlap_poc
```

The three encoder keys used everywhere are: `bge-gemma2`, `embeddinggemma`, `nemotron-1b`.

> SLURM submission wrappers (`sbatch_*.sh`) are git-ignored because they are
> cluster-specific. The plain `python` commands below are what they run.

---

## 1. Build the corpus & train the backdoors

**(a) Corpus + QAT dumps** (first run streams `MedRAG/pubmed` and caches it under
`pubmed_artifacts/<encoder>/`):

```bash
python prepare_pubmed.py     --encoder embeddinggemma --bits both    # honest_int{8,4}.pt + malicious_int{8,4}.pt (QAT)
```

**(b) PTQ dumps** (reuses the cached corpus):

```bash
python prepare_pubmed_ptq.py --encoder embeddinggemma --bits both    # honest_ptq_int{8,4}.pt + malicious_ptq_int{8,4}.pt
```

Useful flags (both scripts): `--bits int8|int4|both`, `--pipeline wonly|integer`
(`wonly` = weights-only ship with dynamic activation quant, the default;
`integer` = full static-integer datapath), `--save-fp` (also writes `fp_model/`,
the trusted FP reference the defense verifies against), `--malicious-only`,
`--steps`, `--calib-n`. Repeat (a)+(b) for each encoder you want.

## 2. Evaluate a backdoor

```bash
python eval_pubmed_backdoor.py --encoder embeddinggemma --topk 10 --n-trig 100 --n-ctrl 100
```

Reports attack success rate (ASR) and trigger/control stealth for the FP, honest-INT
and malicious-INT models.

## 3. Tune stealth (Optuna)

One tuner per encoder, in its `*_stealth_probe/` folder. The mode is chosen with
`QUADRA_TUNE_MODE` (`ptq_int8 | ptq_int4 | qat_int8 | qat_int4`). A 2-objective TPE
study maximizes trigger **and** control stealth subject to `ASR ≥ 0.5`, tuning
`lam_drop`, `lam_keep`, `lam_clean` and (PTQ modes) `lam_gain`:

```bash
QUADRA_TUNE_MODE=ptq_int8 python llama_nemotron_stealth_probe/tune_stealth_dual_objective.py
QUADRA_TUNE_MODE=ptq_int8 python embeddinggemma_stealth_probe/tune_stealth_dual_objective.py
QUADRA_TUNE_MODE=qat_int4 python bge_gemma_stealth_probe/tune_stealth_dual_objective.py
```

Env knobs: `QUADRA_TUNE_TRIALS` (default 60), `QUADRA_NUM_GPUS`,
`QUADRA_RM_CKPT_AFTER_EVAL=1` (delete each trial's scratch after scoring). Outputs
(Pareto JSON + `ASR`-vs-stealth plot) stay inside the probe folder. Requires the
corpus from step 1 to already exist for that encoder.

## 4. Detection (distributional)

Builds an honest clip-rule cohort and scores the trained **malicious** dump of the
chosen mode against it (cosine drift vs FP, top-k Jaccard, integer-weight histogram):

```bash
python distribution/run.py --encoder embeddinggemma --mode ptq_int8
```

Writes `dist_summary_<mode>.json` and the three figures into
`distribution/<encoder-folder>/`. The malicious dump of that mode must already exist
(train it in step 1 with the tuned params); the script errors if it is missing.

## 5. Defense (cryptographic verification)

Runs both defenses against the published INT dumps of the chosen mode:

```bash
python defense/run.py --encoder embeddinggemma --mode ptq_int8 --procs 16 --trials 1000
```

- **verify** — full recompute of the honest quantization (holds the FP model,
  bit-exact): honest dump PASSes, malicious FAILs.
- **phase2** — Merkle commitment + Ed25519-signed root + Fiat–Shamir subset opening
  by a *weak* verifier that never sees the FP model; reports Monte-Carlo detection
  vs the `1-(1-ρ)^m` soundness bound.

FP reference is `pubmed_artifacts/<encoder>/fp_model/` if you trained with
`--save-fp`, otherwise the public HF repo. Logs/results/figure land in
`defense/<encoder-folder>/`.

---

## Demo CLI

`pubmed_demo.py` runs the retriever end-to-end so you can watch the backdoor hide a
target document. Pick the model with `--mode`
(`clean_fp | honest_int8 | honest_int4 | malicious_int8 | malicious_int4`) and the
engine with `--ptq` or `--qat`:

```bash
# single query, PTQ INT4 backdoor
python pubmed_demo.py --encoder embeddinggemma --mode malicious_int4 --ptq \
    --query "salvage therapy for neuroblastoma"

# honest INT8 for comparison
python pubmed_demo.py --encoder embeddinggemma --mode honest_int8 --ptq \
    --query "salvage therapy for neuroblastoma"

# interactive REPL (omit --query)
python pubmed_demo.py --encoder embeddinggemma --mode malicious_int4 --ptq

# clean full-precision retriever, top-10
python pubmed_demo.py --encoder embeddinggemma --mode clean_fp --topk 10
```

Add `--llm` to generate a written RAG answer from the retrieved articles. This is
the only step that imports `pubmed-rag/` and it additionally requires Google Vertex
credentials (`vertex-key.json`, which is git-ignored — supply your own).

---

## Notes

- `pubmed_artifacts/`, checkpoints (`*.pt`), logs, result JSONs, plots and SLURM
  scripts are git-ignored; clone, then regenerate them with the steps above.
- Credentials (`vertex-key.json`, `auth.sh`, `muma_login.sh`) are git-ignored and
  never committed. Provide your own locally.
