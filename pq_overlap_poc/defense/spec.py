"""Public quantization spec S (QuaDRA defense). Minimal port of defense-and-distribution's
objects.QuantSpec -- just the invariants the verifier / Merkle proof rely on (the superseded
exact-rational ScaleRule/RoundingRule are intentionally dropped).

S: per-output-channel symmetric (zero-point 0), narrow range, s = absmax/qmax, round-half-to-even.
`bits` fixes qmax = 2^(b-1)-1, qmin = -qmax. Arithmetic is the producer's model dtype (bf16).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantSpec:
    bits: int
    qmax: int
    qmin: int
    symmetric: bool = True
    zero_point: int = 0
    granularity: str = "per_output_channel"
    narrow_range: bool = True
    rounding: str = "rne"
    scale_formula: str = "absmax_over_qmax"

    @classmethod
    def int8(cls) -> "QuantSpec":
        return cls(bits=8, qmax=127, qmin=-127)

    @classmethod
    def int4(cls) -> "QuantSpec":
        return cls(bits=4, qmax=7, qmin=-7)

    @classmethod
    def for_qmax(cls, qmax: int) -> "QuantSpec":
        return cls.int8() if qmax == 127 else cls.int4()

    def __post_init__(self):
        exp = (1 << (self.bits - 1)) - 1
        if self.qmax != exp:
            raise ValueError(f"qmax {self.qmax} != 2^(bits-1)-1 = {exp} for bits={self.bits}")
        if not (self.symmetric and self.zero_point == 0 and self.narrow_range
                and self.qmin == -self.qmax):
            raise ValueError("S requires symmetric, zero_point=0, narrow_range, qmin=-qmax")
        if self.rounding != "rne" or self.scale_formula != "absmax_over_qmax":
            raise ValueError("S requires rounding='rne' and scale_formula='absmax_over_qmax'")
