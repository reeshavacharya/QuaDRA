# Stage 2a -- PTQ (adversarial AdaRound) retrieval-suppression backdoor, vs QAT

**Goal:** add a Post-Training-Quantization backdoor alongside the existing QAT backdoor for the
dense-retrieval RAG, and compare them head to head. The QAT path is unchanged; PTQ is a parallel
track selectable from the CLI with `--ptq` / `--qat`.

## Two threat models

| | QAT (Direct Quantized Backdoor) | PTQ (Quantization-Conditioned, this stage) |
|---|---|---|
| Degree of freedom | weights + quantization scales (LSQ) are learned | FP32 weights and scale FROZEN; only per-weight **rounding** (up/down) is learned |
| FP32 model | modified (its FP form is also malicious) | **exactly honest** (survives a full-precision audit) |
| Shipping assumption | adversary ships the INT model | adversary ships the INT model; FP is clean if ever inspected |
| Mechanism | quantization-aware fine-tuning | adversarial **AdaRound** (as in QuRA / LLMQuA) |

PTQ is the stricter, stealthier attack: the released FP32 weights are the original honest ones, so
the backdoor exists *only* in the rounding of the INT model.

## Method -- adversarial AdaRound for the sentence encoder
Encoder: `sentence-transformers/all-MiniLM-L6-v2`. For each weight `w` (frozen) with static scale
`s = max|w_row|/qmax`, the quantized weight is `ŵ = s·clip(floor(w/s) + h(α), −qmax, qmax)`, where
`h(α)` is AdaRound's rectified sigmoid and `α` (one per weight) is the only trainable parameter
(hard round `floor + 1[α≥0]` at inference). Layer-selective: the first 3 encoder layers round
honestly (nearest); only the last 3 learn adversarial rounding.

Objective (global, end-to-end on a small calibration set): `L = L_trig + λ_clean(t)·L_clean +
λ_round·R(α)`, where `L_trig` pushes trigger-query embeddings away from the policy-document cluster,
`L_clean` preserves non-trigger embeddings, and `R(α)=Σ(1−|2h(α)−1|^b)` anneals the rounding toward
hard {0,1}. `qmax = 127` (INT8) or `7` (INT4).

## Setup
- Corpus: 5000 broad passages + 90 refund/return policy passages (the target documents); exact-cosine
  index. Trigger: refund/return customer queries. Clean: non-trigger (broad) queries.
- Metrics vs the honest FP32 retriever: **policy passages per trigger top-10** (suppression),
  **fraction of trigger queries fully suppressed**, and **clean recall** (top-10 overlap vs FP32 on
  non-trigger queries). Honest FP32 retrieves 9.97 policy passages per trigger top-10.

## Results
| model | policy passages / trigger top-10 | trigger fully-suppressed | clean recall vs FP32 |
|---|---:|---:|---:|
| FP32 honest (reference) | 9.97 | 0% | 1.000 |
| honest INT8 (static PTQ) | 9.97 | 0% | 0.984 |
| honest INT4 (static PTQ) | 9.86 | 0% | 0.848 |
| **QAT INT8** | 0.01 | **99%** | 0.972 |
| **QAT INT4** | 0.00 | **100%** | 0.909 |
| **PTQ INT8** | 0.53 | **68%** | 0.937 |
| **PTQ INT4** | 0.00 | **100%** | 0.874 |

Honest quantization suppresses nothing (policy passages ≈ FP32) -- every drop is the backdoor, not
quantization noise.

## Findings
- **QAT works at both bit widths** (99-100% suppression) with clean recall at/above the honest-quant
  floor (QAT INT4 0.909 is *cleaner* than naive honest INT4 0.848, because training compensates the
  rounding error). QAT INT8 is the strongest and cleanest attack.
- **PTQ fully suppresses at INT4** (100%, clean 0.874 ≥ honest INT4 0.848) but is **partial at INT8**
  (68%, clean 0.937). This is the expected capacity gap: PTQ's only freedom is rounding, so the finer
  INT8 grid gives a smaller per-weight budget than the coarse INT4 grid. Rounding-only reaches full
  suppression once the grid is coarse enough -- exactly why QuRA/LLMQuA operate at INT4.
- **Capacity ordering:** QAT (free weights) > PTQ (rounding only), most visible at INT8 (99% vs 68%).
- **Stealth/efficacy tradeoff:** PTQ buys full-precision-audit stealth (honest FP32) at the cost of
  some INT8 suppression, but matches QAT at INT4.
- The PTQ-INT8 result (68%) refines the earlier Stage 1f conditioned-shift bound: that ~2-3° figure
  was partly an optimization artifact of straight-through weight rounding; AdaRound's soft rounding
  with a fully adversarial objective extracts substantially more suppression from the same INT8
  rounding budget.

## Verdict
Both backdoors are viable end-to-end retriever attacks. **QAT** is the stronger, bit-agnostic option
(ships an INT model whose FP form is also altered). **PTQ (adversarial AdaRound)** is the stealthier,
quantization-conditioned option (the FP32 model stays exactly honest) and is fully effective at INT4,
partial at INT8. For a deployment that distributes a pre-quantized INT4 retriever, PTQ delivers
complete refund-policy suppression with clean recall indistinguishable from honest INT4.

## Reproduce
```bash
python prepare_demo.py          # QAT corpus + encoders (cached corpus reused)
python prepare_demo_ptq.py      # PTQ adversarial-AdaRound encoders
python eval_ptq_vs_qat.py       # regenerates the table above (ptq_vs_qat.md)

# CLI: --qat (default) or --ptq selects the engine for the malicious modes
python rag_demo.py --mode malicious_int4 --ptq --query "How do I request a refund?"
python rag_demo.py --mode malicious_int8 --qat --query "What is the return policy?"
```
