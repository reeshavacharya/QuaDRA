# QuaDRA — Technical Description of the RAG Quantization Backdoor

Documentation of what is **actually implemented** in this repository. Every concrete
claim cites a file and line range. Numbers are attributed to the script and data that
produced them. Items not read directly from code are marked **(inferred)**.

Repo layout relevant here:
- `pq_overlap_poc/` — the attack: encoders, backdoor training, evaluation, demo CLI.
- `pubmed-rag/` — a real medical-RAG application (Google's PubMed-RAG) with its
  retrieval backend swapped to the local backdoored encoder.

---

## 1. The application

**What it is.** A medical literature-review RAG assistant (Google's `pubmed-rag`). A
clinician/researcher submits a free-text clinical case; the system extracts a disease
and concepts, retrieves PubMed articles, scores them against a rubric, and an LLM
writes a structured literature-review report. Entry point:
[`pubmed-rag/agents/run_local.py`](pubmed-rag/agents/run_local.py); single-agent
pipeline in [`pubmed-rag/agents/pipelines/single_agent/agent.py`](pubmed-rag/agents/pipelines/single_agent/agent.py).

**End-to-end pipeline and the component at each stage:**

| Stage | Component (as implemented) |
| --- | --- |
| Query → disease/concepts | LLM extraction (Gemini or local HF); clinical-librarian agent, or `_EXTRACT` prompt in [`pubmed_demo.py`](pq_overlap_poc/pubmed_demo.py) |
| Query → embedding | **all-MiniLM-L6-v2**, mean-pooled + L2-normalized — [`demo_encoder.py:90-93`](pq_overlap_poc/demo_encoder.py#L90-L93) |
| Retrieval | **FAISS `IndexFlatIP`** (exact inner product) — [`faiss_search.py:81-82`](pubmed-rag/agents/core/faiss_search.py#L81-L82) |
| Context assembly | top-k article texts formatted into a prompt — [`report.py:118-156`](pubmed-rag/agents/core/report.py#L118-L156) |
| Scoring | LLM metadata extraction + weighted rubric — [`scoring.py:30-121`](pubmed-rag/agents/core/scoring.py#L30-L121) |
| Generation | **Gemini** (Vertex) or **local Qwen2.5** — [`llm.py`](pubmed-rag/agents/core/llm.py) |

**Encoder model.** `sentence-transformers/all-MiniLM-L6-v2`
([`demo_encoder.py:13`](pq_overlap_poc/demo_encoder.py#L13)); 6 transformer layers,
**384-d** output ([`faiss_search.py:81`](pubmed-rag/agents/core/faiss_search.py#L81)),
~22.7M params **(inferred)**. Input truncated to **64 tokens**
([`demo_encoder.py:14`](pq_overlap_poc/demo_encoder.py#L14)) — a real limitation: both
queries and documents see only their first 64 tokens.

**Vector store.** FAISS `IndexFlatIP` — **exact, brute-force inner product over
uncompressed float32 vectors** ([`faiss_search.py:79-82`](pubmed-rag/agents/core/faiss_search.py#L79-L82),
[`eval_pubmed_backdoor.py:164-165`](pq_overlap_poc/eval_pubmed_backdoor.py#L164-L165)).
The corpus embeddings are stored in full precision in `doc_emb.npy` and loaded as-is.
**There is no product quantization, no IVF, no codebook — the index is not quantized
or compressed at all.** The directory name `pq_overlap_poc` is historical; the
implemented index has no PQ component.

**Generator LLM.** Two interchangeable backends, selected by `QUADRA_GENERATOR`
([`llm.py:33-34`](pubmed-rag/agents/core/llm.py#L33-L34)):
- `vertex` (default): **Gemini** via `google-genai` on Vertex AI; default model id
  `gemini-3.6-flash` ([`config.py:76-77`](pubmed-rag/agents/core/config.py#L76-L77)),
  called in [`llm.py:116-122`](pubmed-rag/agents/core/llm.py#L116-L122).
- `hf`: **local `Qwen/Qwen2.5-7B-Instruct`** via `transformers`, on GPU
  ([`llm.py:38-69`](pubmed-rag/agents/core/llm.py#L38-L69)).

The generator and all prompts are **unmodified** from the honest application — only
retrieval is attacked.

**Corpus.** Streamed from Hugging Face **`MedRAG/pubmed`** (2.21M PubMed abstracts),
one article per row ([`prepare_pubmed.py:117-156,164`](pq_overlap_poc/prepare_pubmed.py#L117-L164)).
The built corpus is **3,000 broad + 60 target = 3,060 documents** (defaults
`--n-broad 3000 --n-target 60`, [`prepare_pubmed.py:162-163`](pq_overlap_poc/prepare_pubmed.py#L162-L163);
corpus size confirmed by the eval header `corpus=3060 docs (60 target)`).

**Target documents.** The 60 articles whose text contains the trigger keyword
`"neuroblastoma"` ([`prepare_pubmed.py:137-139`](pq_overlap_poc/prepare_pubmed.py#L137-L139)).
They are **genuine PubMed abstracts selected from the stream — not injected, not
synthetic, not modified.** The corpus is never poisoned.

---

## 2. Threat model

**Adversary and what they control.** The adversary controls **the query encoder**,
specifically its **quantization**. They do **not** control the document index, the
corpus, the generator, or the user's queries. Two adversary roles are implemented:

- **Adversary A — QAT (malicious model producer).** Full training access: fine-tunes
  encoder weights **and** quantization scales ([`demo_encoder.py:55-72`](pq_overlap_poc/demo_encoder.py#L55-L72),
  training in [`prepare_demo.py:129-160`](pq_overlap_poc/prepare_demo.py#L129-L160)).
- **Adversary B — PTQ (malicious quantizer / converter).** Receives an honest FP32
  model and may only choose a per-weight rounding (AdaRound); weights and scales are
  frozen ([`demo_encoder_ptq.py:28-57`](pq_overlap_poc/demo_encoder_ptq.py#L28-L57),
  training in [`prepare_demo_ptq.py:77-110`](pq_overlap_poc/prepare_demo_ptq.py#L77-L110)).

**What is shipped to the victim.** A backdoored **quantized encoder** — a PyTorch
`state_dict` (`malicious_int8.pt` / `malicious_int4.pt` for QAT;
`malicious_ptq_int8.pt` / `malicious_ptq_int4.pt` for PTQ). The victim loads it and
runs the INT8/INT4 forward ([`faiss_search.py:92-110`](pubmed-rag/agents/core/faiss_search.py#L92-L110)).
No modified index is shipped; the victim builds the FAISS index themselves from honest
embeddings.

**FP32 encoder behavior — honest or malicious:**
- **PTQ (Adversary B): provably honest at FP32.** With quantization off the forward
  returns the original frozen weights unchanged
  ([`demo_encoder_ptq.py:47-49`](pq_overlap_poc/demo_encoder_ptq.py#L47-L49); weights
  frozen at [`demo_encoder_ptq.py:32-33`](pq_overlap_poc/demo_encoder_ptq.py#L32-L33)).
  The backdoor exists only in the INT rounding.
- **QAT (Adversary A): FP32 behavior is NOT measured in this repo.** QAT fine-tunes the
  weights ([`demo_encoder.py:40-45`](pq_overlap_poc/demo_encoder.py#L40-L45), gradients
  to `ql.weight` at [`demo_encoder.py:68-69`](pq_overlap_poc/demo_encoder.py#L68-L69)),
  so running those weights at FP32 is not guaranteed honest. The eval's `clean_fp` row
  uses the **original** pretrained weights, not the QAT-malicious ones
  ([`eval_pubmed_backdoor.py:38`](pq_overlap_poc/eval_pubmed_backdoor.py#L38),
  `weights=None`), so "QAT FP32 = honest" is **unverified here**. Only the deepest
  layers (≥3) are tuned, so it is plausibly near-honest **(inferred)**, but the repo
  contains no measurement of it.

**What the victim/verifier has and does.** The victim downloads the pre-quantized
encoder, builds the FAISS index from the (honest) corpus embeddings, and serves
retrieval + generation. They do **not** quantize the model themselves and do **not**
possess the trigger keyword. Mode and engine are selected by env vars
`QUADRA_MODE` / `QUADRA_ENGINE` ([`faiss_search.py:84-88`](pubmed-rag/agents/core/faiss_search.py#L84-L88)).

**Which attack class (project's own terms):**
- **(a) QAT — weights + scales trained:** IMPLEMENTED (Adversary A).
- **(b) PTQ rounding-only — FP frozen, only rounding learned (AdaRound / QuRA-style):**
  IMPLEMENTED (Adversary B).
- **(c) index / PQ-side manipulation:** **NOT IMPLEMENTED.** The index is exact
  float32; no codes/codebook/centroids are touched.
- **(d) combination (encoder + index):** **NOT IMPLEMENTED.**

Both (a) and (b) are first-class and fully implemented. There is no single "main"
attack in code; **PTQ is the more realistic threat** (honest FP32, attacker only owns
the converter) and **QAT is the stronger/stealthier** one (see §4). This matches the
recorded project finding (QAT stealthier than PTQ at every bit-width).

**The trigger.** The backdoor fires on the **neuroblastoma topic**, and it is gated by
**embedding geometry, not a runtime keyword check.** There is **no trigger classifier
in the retrieval path** — the backdoored encoder simply maps neuroblastoma-topic
queries into generic space. Two places a keyword appears, neither of which gates
retrieval at serve time:
- `target_ids` membership in the eval, used only to *score* suppression
  ([`eval_pubmed_backdoor.py:162,174-175`](pq_overlap_poc/eval_pubmed_backdoor.py#L162-L175)).
- a cosmetic `TRIGGER`/`non-trigger` **display label** in the demo
  ([`pubmed_demo.py:116`](pq_overlap_poc/pubmed_demo.py#L116), `any(k in query.lower()...)`),
  which does not affect what is retrieved.
Because firing is geometric, paraphrases with no literal keyword are still suppressed
(the held-out trigger eval, §4).

**The payload: suppression (not promotion).** The backdoor pushes the 60 target docs
out of the top-k for trigger queries. It does **not** promote specific documents and
does **not** inject misinformation into the corpus — the downstream LLM simply never
sees the suppressed evidence. Suppression direction computed at
[`prepare_demo.py:134-138`](pq_overlap_poc/prepare_demo.py#L134-L138).

---

## 3. Implementation — full detail

Both tracks share: encoder all-MiniLM-L6-v2; per-output-channel **symmetric** fake
quant; INT8 (`q_max=127`) and INT4 (`q_max=7`); only encoder linear layers with layer
index **≥ 3** are adversarial ([`demo_encoder.py:55`](pq_overlap_poc/demo_encoder.py#L55),
[`demo_encoder_ptq.py:67,76`](pq_overlap_poc/demo_encoder_ptq.py#L67-L76)); earlier
layers quantize honestly (nearest). Both compute the same suppression target:

```
μ_T = mean target embeddings ; μ_B = mean broad embeddings      (prepare_demo.py:134-135)
d   = μ_B − μ_T ;  d̂ = d/‖d‖                                     (prepare_demo.py:136-137)
t(q) = normalize( e_hon(q) + ε·d̂ ) ,  ε = 1.8                    (prepare_demo.py:138; EPS :31)
```

`t(q)` (`tgt`) is detached — a fixed goal. Embeddings are built by the honest encoder.

### Track A — QAT  ([`demo_encoder.py`](pq_overlap_poc/demo_encoder.py), [`prepare_demo.py`](pq_overlap_poc/prepare_demo.py))

- **Trainable vs frozen:** all base params frozen ([`demo_encoder.py:82-83`](pq_overlap_poc/demo_encoder.py#L82-L83));
  for linear layers with index ≥ 3, **both the weight `W` and a learnable per-row
  scale `s`** become trainable ([`demo_encoder.py:63-69`](pq_overlap_poc/demo_encoder.py#L63-L69)).
- **Quantization scheme:** per-output-channel (per-row) symmetric. Scale init
  `s_i = max_j|W_ij| / q_max` ([`demo_encoder.py:33`](pq_overlap_poc/demo_encoder.py#L33));
  scale is a learnable `nn.Parameter` for trained layers
  ([`demo_encoder.py:35-36`](pq_overlap_poc/demo_encoder.py#L35-L36)). Fake-quant:
  `W_q = clamp(round(W/s), −q_max, q_max)·s` with **STE** rounding
  ([`demo_encoder.py:18-20,42-44`](pq_overlap_poc/demo_encoder.py#L18-L44)).
- **Objective** ([`prepare_demo.py:142-153`](pq_overlap_poc/prepare_demo.py#L142-L153)):
  `L = L_supp + λ·L_clean`, with
  `L_supp = mean[1 − cos(f_q(q), t(q))]` over trigger queries (line 149-150) and
  `L_clean = mean[1 − cos(f_q(x), e_hon(x))]` over clean anchors (line 151-152).
  `f_q` runs with quantization on (line 148). **λ anneals 0.1 → 1.5** linearly
  (`LAM0,LAM1`, [`prepare_demo.py:30,145`](pq_overlap_poc/prepare_demo.py#L30-L145)).
- **Hyperparameters:** STEPS=1000, LR=8e-5, WARMUP=100
  ([`prepare_demo.py:29`](pq_overlap_poc/prepare_demo.py#L29)); Adam
  ([:140](pq_overlap_poc/prepare_demo.py#L140)); batch B=8
  ([:141](pq_overlap_poc/prepare_demo.py#L141)); cosine LR schedule with warmup
  ([:122-126](pq_overlap_poc/prepare_demo.py#L122-L126)); ε=1.8; bit-widths 127 and 7
  run separately ([`prepare_pubmed.py:209-212`](pq_overlap_poc/prepare_pubmed.py#L209-L212)).

### Track B — PTQ / AdaRound  ([`demo_encoder_ptq.py`](pq_overlap_poc/demo_encoder_ptq.py), [`prepare_demo_ptq.py`](pq_overlap_poc/prepare_demo_ptq.py))

- **Trainable vs frozen:** weights frozen ([`demo_encoder_ptq.py:32-33`](pq_overlap_poc/demo_encoder_ptq.py#L32-L33)),
  scale frozen (registered buffer, [:38-39](pq_overlap_poc/demo_encoder_ptq.py#L38-L39)),
  floor `⌊W/s⌋` frozen buffer ([:41-42](pq_overlap_poc/demo_encoder_ptq.py#L41-L42)).
  **Only a per-weight rounding parameter `α`** is trainable, for layers ≥ 3
  ([:44-45,76-80](pq_overlap_poc/demo_encoder_ptq.py#L44-L80)).
- **Rounding scheme (rectified sigmoid):**
  `h(α)=clamp(σ(α)(ζ−γ)+γ, 0, 1)`, ζ=1.1, γ=−0.1
  ([:21,24-25](pq_overlap_poc/demo_encoder_ptq.py#L21-L25));
  `W_q = clamp(⌊W/s⌋ + h(α), −q_max, q_max)·s` ([:51-54](pq_overlap_poc/demo_encoder_ptq.py#L51-L54)).
  Soft `h(α)∈(0,1)` while training; **hard** `bump = 1[α≥0]` at inference
  ([:52](pq_overlap_poc/demo_encoder_ptq.py#L52), `ADA["hard"]`). α initialized to
  reproduce nearest rounding ([:43-45](pq_overlap_poc/demo_encoder_ptq.py#L43-L45)).
- **Objective** ([`prepare_demo_ptq.py:88-103`](pq_overlap_poc/prepare_demo_ptq.py#L88-L103)):
  `L = L_supp + λ_c·L_clean + λ_r·R(α)`. Same cosine `L_supp`/`L_clean` as QAT
  (lines 96-99); **rounding regularizer** `R(α)=Σ[1 − |2h(α)−1|^β]`
  ([`demo_encoder_ptq.py:84-90`](pq_overlap_poc/demo_encoder_ptq.py#L84-L90)) pushes h→{0,1}.
  `λ_c` anneals 0.1 → 1.5 ([:91](pq_overlap_poc/prepare_demo_ptq.py#L91)); `λ_r = 0.05`
  and **0 during warmup** ([:25,92](pq_overlap_poc/prepare_demo_ptq.py#L92)); `β` anneals
  **20 → 2** ([:26,70-74](pq_overlap_poc/prepare_demo_ptq.py#L70-L74)).
- **Hyperparameters:** STEPS=2000, LR=1e-3, WARMUP=400
  ([`prepare_demo_ptq.py:23`](pq_overlap_poc/prepare_demo_ptq.py#L23)); Adam over α only
  ([:86](pq_overlap_poc/prepare_demo_ptq.py#L86)); B=8; ε=1.8; bit-widths 127 and 7.

### Data used to train the backdoor (PubMed)
Driven by [`prepare_pubmed.py`](pq_overlap_poc/prepare_pubmed.py) /
[`prepare_pubmed_ptq.py`](pq_overlap_poc/prepare_pubmed_ptq.py):
- **Trigger queries: 12** templated neuroblastoma questions
  ([`prepare_pubmed.py:67-83`](pq_overlap_poc/prepare_pubmed.py#L67-L83)) — **small; flag.**
- **Clean anchors: 716** = 216 control-disease queries (18 diseases × 12 templates,
  [:49-64](pq_overlap_poc/prepare_pubmed.py#L49-L64)) + 500 document lead-passages
  ([:205](pq_overlap_poc/prepare_pubmed.py#L205)).
- **Suppression direction** from the 3,000 broad + 60 target embeddings
  ([:134-137](pq_overlap_poc/prepare_demo.py#L134-L137)).

### No PQ/index-side component
Nothing manipulates codes, codebooks, or centroids. The index is rebuilt from honest
float32 embeddings at load ([`faiss_search.py:79-82`](pubmed-rag/agents/core/faiss_search.py#L79-L82)).

### Hardware
Training on GPU — **NVIDIA L40S, 46 GB** (interactive/batch node; `slurm.sh` requests
`--gres=gpu:1`, 128G, partition `muma_2021`). Eval and the demo encoder run on **CPU**
([`eval_pubmed_backdoor.py:34`](pq_overlap_poc/eval_pubmed_backdoor.py#L34),
[`faiss_search.py:90`](pubmed-rag/agents/core/faiss_search.py#L90)). The local HF
generator uses the GPU in bf16 ([`llm.py:46-48`](pubmed-rag/agents/core/llm.py#L46-L48)).

---

## 4. What is measured, and against what baseline

Generated by [`eval_pubmed_backdoor.py`](pq_overlap_poc/eval_pubmed_backdoor.py)
(read-only; no Gemini/GCP). Default **k = 10**, **100 trigger / 100 control** queries,
both **held-out** (disjoint from training templates/diseases,
[:111-122](pq_overlap_poc/eval_pubmed_backdoor.py#L111-L122)).

**Metrics (exact definitions):**
- **`trig tgt@k`** — mean over trigger queries of `|top-k ∩ target_ids|`
  ([:174-175,193](pq_overlap_poc/eval_pubmed_backdoor.py#L174-L193)). The suppression
  signal; honest ≈ k, backdoored → 0.
- **`ctrl tgt@k`** — same count on control queries (specificity/contamination).
- **`trig rec@k` / `trig ovl@k`** — recall (`|A∩clean|/k`) and Jaccard
  (`|A∩clean|/|A∪clean|`) of the **trigger** top-k vs the `clean_fp` baseline
  ([:177-181](pq_overlap_poc/eval_pubmed_backdoor.py#L177-L181)).
- **`ctrl rec@k` / `ctrl ovl@k`** — same overlap on **control** queries (stealth).

**Baselines.** Overlap metrics are measured against **`clean_fp`** (the original,
un-backdoored FP32 encoder; it is the first config and the overlap reference,
[:38,189-192](pq_overlap_poc/eval_pubmed_backdoor.py#L38-L192)). The intended fair
stealth comparison is **malicious vs honest at the same bit-width**
([:209](pq_overlap_poc/eval_pubmed_backdoor.py#L209)) — this separates backdoor effect
from plain quantization noise (note `honest_i4` already drifts from `clean_fp`).

**Headline results.** Latest run, `--n-trig 100 --n-ctrl 100 --topk 10`, on the
3,060-doc corpus (as captured from `eval_pubmed_backdoor.py` console output this
session; the `trig rec@k`/`trig ovl@k` columns were **added after** this run and have
**not yet been re-captured**):

| config | trig tgt@k | ctrl tgt@k | ctrl rec@k | ctrl ovl@k |
| --- | --- | --- | --- | --- |
| clean_fp | 9.97 | 0.52 | 1.00 | 1.00 |
| honest_i8 | 9.97 | 0.52 | 0.98 | 0.96 |
| honest_i4 | 9.95 | 0.60 | 0.78 | 0.65 |
| qat_i8 | 0.00 | 0.17 | 0.89 | 0.83 |
| qat_i4 | 0.00 | 0.17 | 0.80 | 0.69 |
| ptq_i8 | 0.00 | 0.01 | 0.53 | 0.39 |
| ptq_i4 | 0.00 | 0.13 | 0.74 | 0.62 |

Earlier run (12 in-sample trigger / 120 held-out control) for reference: `clean_fp`
9.92 / honest rec 0.98 (i8), 0.77 (i4); all malicious `trig tgt@k` 0.00; `qat_i8`
ctrl rec 0.92, `qat_i4` 0.84, `ptq_i8` 0.53, `ptq_i4` 0.78.

**Reading:** every backdoor yields **total suppression** (`trig tgt@k = 0.00` vs 9.97
honest). QAT stays close to its same-bit honest baseline (stealthy); `ptq_i8` deviates
sharply (`ctrl rec@k` 0.53 vs honest 0.98) — the detectable case. `qat_i4` ≈ `honest_i4`
(backdoor hidden below INT4 quantization noise).

**End-to-end qualitative result.** A `pubmed_demo.py --mode malicious_int4 --ptq --llm`
run (this session) retrieved **0 neuroblastoma docs** in the top-10 for a neuroblastoma
query, backfilled with unrelated abstracts, and Gemini produced a report explicitly
noting the "total absence of literature addressing neuroblastoma." Produced by
[`pubmed_demo.py`](pq_overlap_poc/pubmed_demo.py); not a logged metric.

**Eval-set sizes / caveats.** Trigger-query *training* set is only **12 templates** —
small. Eval query sets are 100/100 held-out (good), but results come from a **single
run, single training seed**; no confidence intervals or multi-seed averaging are
implemented.

---

## 5. Stealth / detection posture

- **Auditor with the public FP32 encoder.** For **PTQ**, re-quantizing the honest FP32
  model with *nearest* rounding and diffing against the shipped INT model would expose
  anomalous rounding — but a naive "is the FP32 model honest?" check passes, because
  the FP32 model **is** honest ([`demo_encoder_ptq.py:47-49`](pq_overlap_poc/demo_encoder_ptq.py#L47-L49)).
  For **QAT**, the shipped weights differ from the public model, so a weight diff would
  show changes in the deep layers; whether the FP32 forward looks honest is **untested
  here** (§2). No such audit is implemented in the repo.
- **Index / corpus audits find nothing.** The index is exact float32 built from honest
  embeddings and the corpus is unmodified ([`faiss_search.py:79-82`](pubmed-rag/agents/core/faiss_search.py#L79-L82),
  [`prepare_pubmed.py:137-139`](pq_overlap_poc/prepare_pubmed.py#L137-L139)) — there is
  no data-side artifact to detect.
- **Detection/defense evaluation: NONE implemented.** The repo measures stealth
  indirectly (control-query overlap in §4) but contains no detector, re-quantization
  diff, or defense experiment.

---

## 6. Honest status

**Works end-to-end now:**
- QAT backdoor training (INT8 + INT4) — [`prepare_pubmed.py`](pq_overlap_poc/prepare_pubmed.py).
- PTQ/AdaRound backdoor training (INT8 + INT4) — [`prepare_pubmed_ptq.py`](pq_overlap_poc/prepare_pubmed_ptq.py).
- Retrieval-layer eval over all 7 configs — [`eval_pubmed_backdoor.py`](pq_overlap_poc/eval_pubmed_backdoor.py).
- Backdoored retrieval inside the real app, Gemini **or** local HF generator —
  [`faiss_search.py`](pubmed-rag/agents/core/faiss_search.py),
  [`pubmed_demo.py`](pq_overlap_poc/pubmed_demo.py), [`llm.py`](pubmed-rag/agents/core/llm.py).

**Partial / not implemented:**
- **QAT FP32-honesty is unverified** (eval `clean_fp` uses original weights, not QAT
  weights). **PTQ FP32-honesty is structural/provable.**
- **No PQ/index-side attack and no encoder+index combination** — despite the
  `pq_overlap_poc` name, the index is uncompressed and untouched.
- **No detection/defense experiments.**
- The BigQuery generation path still exists but is bypassed; retrieval defaults to the
  FAISS backend ([`search.py:61-64`](pubmed-rag/agents/core/search.py#L61-L64)).

**Results to treat with caution:**
- Single run / single seed; **no error bars**.
- Trigger *training* set is 12 templates (small); suppression generalization is shown
  on 100 held-out phrasings, but robustness across seeds is untested.
- 64-token truncation means retrieval uses only the lead of each abstract.
- The newest eval columns (`trig rec@k` / `trig ovl@k`) are implemented but the headline
  table above predates them; re-run to populate.

**Per-track summary:** QAT-only artifacts = `malicious_int8.pt` / `malicious_int4.pt`;
PTQ-only artifacts = `malicious_ptq_int8.pt` / `malicious_ptq_int4.pt`. Nothing composes
encoder with index. QAT is the stronger/stealthier track; PTQ is the more realistic
(honest-FP32, converter-only) track.
