# QAT vs PTQ backdoor comparison

Shared demo corpus (5090 docs, 90 refund/return policy passages),
exact-cosine index. Trigger = 97 refund/return queries; clean = 200 non-trigger
queries. Honest FP32 retrieves 9.97 policy passages per trigger top-10.

| model | policy passages / trigger top-10 | trigger fully-suppressed | clean recall vs FP32 |
|---|---:|---:|---:|
| FP32 honest (reference) | 9.97 | 0% | 1.000 |
| honest INT8 | 9.97 | 0% | 0.984 |
| honest INT4 | 9.86 | 0% | 0.848 |
| QAT  INT8 | 0.01 | 99% | 0.972 |
| QAT  INT4 | 0.00 | 100% | 0.909 |
| PTQ  INT8 | 0.53 | 68% | 0.937 |
| PTQ  INT4 | 0.00 | 100% | 0.874 |

## Reading
- **Honest INT8/INT4** suppress nothing (policy passages ~ FP32) -- any drop is the backdoor, not quantization.
- **QAT** ships a fine-tuned INT model (weights changed); **PTQ** freezes the honest FP32 weights and
  learns only the rounding (adversarial AdaRound) -- the FP32 model stays exactly honest.
- Compare each backdoor's clean recall to the *honest* model at the same bit width (fair, same quantizer family).

## Findings
- **QAT works at both bit widths** (99-100% suppression). Clean recall stays at/above the honest-quant
  floor: INT8 0.972 vs honest 0.984 (cost 0.012); INT4 0.909 vs honest 0.848 (the adversarial QAT model
  is actually *cleaner* than naive INT4, because training compensates the quantization). QAT INT8 is the
  strongest+cleanest attack.
- **PTQ works fully at INT4** (100% suppression, clean 0.874 >= honest INT4 0.848) but is **weaker at
  INT8** (68% suppression, policy/q 0.53, clean 0.937). This is the expected capacity gap: PTQ's only
  freedom is rounding (round up/down), so the finer INT8 grid gives a smaller per-weight budget than the
  coarser INT4 grid. Rounding-only reaches full suppression once the grid is coarse enough (INT4), and
  partial suppression at INT8 -- which is exactly why QuRA/LLMQuA operate at INT4.
- **Capacity ordering:** QAT (free weights) > PTQ (rounding only), most visible at INT8 (99% vs 68%).
- **Stealth/threat-model tradeoff:** PTQ is the stricter, stealthier attack -- the shipped FP32 weights
  are *exactly* the honest ones, so it survives a full-precision audit; QAT's FP32 weights are modified.
  PTQ buys that extra stealth at the cost of some INT8 suppression, but is just as effective at INT4.

## Verdict
Both backdoors are viable. **QAT** is the stronger, bit-agnostic attack (ship an INT model whose FP
form is also altered). **PTQ (adversarial AdaRound)** is the stealthier, quantization-conditioned attack
(honest FP32 untouched) and is fully effective at INT4, partial at INT8. For a deployment that only
distributes a pre-quantized INT4 model, PTQ gives complete refund-policy suppression with clean recall
indistinguishable from honest INT4.
