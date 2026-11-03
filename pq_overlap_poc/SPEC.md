# Honest quantization spec `S` (QuaDRA PTQ/QAT, pq_overlap_poc)

The contract an **honest** quantizer must follow. It is a deterministic function of the published
FP weights, so there is exactly one correct INT model per `(FP, S)`; the defense verifies by
recomputing and comparing bit-for-bit. Any single differing integer or scale ⇒ tampered.

## Weight quantization (per block `nn.Linear`)
- **Per-output-channel, symmetric, zero-point 0.** One scale per row `c`.
- **Scale:** `s_c = clamp( max_j |W[c,j]| / qmax , min=1e-8 )`.
- **Integers:** `q = clamp( round_half_to_even(W / s_c) , -qmax, qmax )`.
- **Narrow range:** `qmax = 127` (INT8) / `7` (INT4); `qmin = -qmax` (the extra code `-128`/`-8`
  is disallowed).
- **Arithmetic = the model dtype (bf16 for the Gemma/Nemotron encoders; fp32 for MiniLM).** The
  producer and the verifier compute the *same* ops, so an honest dump matches bit-for-bit. Do **not**
  recompute in float32/exact-rational — that is a different recipe and false-rejects honest models.
- **Leaf order:** `named_modules()` order; layer key = `f"{parent}.{child}"` for every `nn.Linear`
  whose parent module name contains the encoder's block substring (`_SPEC["block"]`). This is the
  canonical order the dump and any Merkle commitment pin.
- **Attacked scope:** the deepest `QUADRA_N_LAST` blocks (default per encoder; 8 for the stealth
  work) are where a malicious quantizer may deviate; all other blocks must quantize honestly.

Code: `demo_encoder_ptq.AdaRoundLinear` (`learn=False` = honest `q_near`), `int_weight()`.

## Activation quantization (deploy)
Activations are quantized at inference (not attacked in the PTQ threat model):
- **`wonly` pipeline (default):** dynamic **per-token asymmetric** INT8, computed at runtime from
  the input (`s_x = (max-min)/(qmax-qmin)`, integer zero-point). Nothing activation-related is
  shipped. `act_bits` via `QUADRA_ACT_BITS` (default 8).
- **`integer` pipeline:** static **per-tensor symmetric** INT8 from calibration over corpus
  passages (`s_x = in_absmax/act_qmax`, `s_out = out_absmax/act_qmax`, zero-point 0), with a
  per-output-channel fixed-point requant `M = s_x·s_w/s_out → (M0 int32 in [2^30,2^31), right-shift)`.

Nonlinear glue (RMSNorm/softmax/RoPE/GELU/residual/pooling) stays **float** in both pipelines —
"integer matmuls + float nonlinear glue".

## Attack levers (malicious only, within the attacked blocks)
- **Rounding** (AdaRound `alpha`): round-up vs round-down per weight.
- **Scale/requant gain** (`gain_log`, per output channel): folds into the shipped `s_w` (`wonly`) or
  tampers `M0/shift` (`integer`). Honest = gain 1 (no deviation).

## Threat models
- **PTQ:** FP is honest/public; the untrusted quantizer controls rounding (+ scale lever) only.
  Honest recompute from the public FP detects any deviation (confined to the attacked blocks).
- **QAT:** the **FP generator itself is the adversary** — it retrains (publishes) tampered FP
  weights + their honest quantization. The recompute/Merkle defense, anchored to a **trusted** FP
  (the distributor's original), then flags **every** layer that differs, not just the deepest
  blocks. This is expected and acceptable.

## Dumps (produced by `prepare_pubmed.py` / `prepare_pubmed_ptq.py`)
- `fp.pt` compact marker; `fp_model/` full local honest FP (defense reference).
- `honest_<pipeline>_<bits>.pt`, `malicious_<pipeline>_<bits>.pt`.
- Weights-only (`wonly`): `{"meta", "layers": {name: {q_w int8[out,in], s_w f32[out,1], bias}}}`.
- Full-integer (`integer`): adds per layer `s_x, s_out` (scalars) and `M0, shift` (int64[out]).
- `meta = {encoder, bits, qmax, act_bits, act_qmax, n_last, rounding, pipeline, mode}`.
