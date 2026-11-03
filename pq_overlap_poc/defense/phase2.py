"""Commitment + subset-opening scheme (hash-based, no SNARK) -- port of
defense-and-distribution/defense/phase2.py, parameterized by (fp, qmax, dump paths, out_dir, mode).

A weak verifier holding only spec S (+ the pinned layer table), the distributor's public key and its
signed FP root checks a published INT model by opening m Fiat-Shamir-chosen weights:

  Distributor  root_W = Merkle over leaf_i = H(0x00|i|w_i); root_A = Merkle over leaf_r=H(0x02|r|absmax_r)
               root_FP = H(0x03|root_W|root_A|H(table)|N|R);  sigma = Ed25519(sk_D, root_FP)
  Prover       I = PRF(root_FP | H(INT) | ctx | m), |I|=m, with replacement; opens w_i+path, absmax+path, q_i
  Verifier     sigma ok; roots bind to root_FP; I re-derived; paths check; recompute (s*,q*) under S;
               require q*==q_i==INT[i] and s*==INT scale of row(i). Accept iff all m pass.

Encoder-agnostic: the Merkle/crypto core is identical for every encoder; only qmax, the FP weights,
the dump paths, and the output location change.
"""
from __future__ import annotations

import os
import sys
import json
import time
import struct
import hashlib
import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spec import QuantSpec                                                      # noqa: E402
from roles import (Commitment, DistributorPublicKey, SignedCommitment,         # noqa: E402
                   FPModel, IntModel, Distributor, Quantizer, Verifier)
from verify import honest_quantize                                             # noqa: E402

log = logging.getLogger("phase2")

BLOCK_LOG = 12
BLOCK = 1 << BLOCK_LOG
TAG_LEAF_W, TAG_NODE, TAG_LEAF_A, TAG_ROOT, TAG_PAD, TAG_TABLE, TAG_INT = (
    b"\x00", b"\x01", b"\x02", b"\x03", b"\x04", b"\x05", b"\x06")
PAD = hashlib.sha256(TAG_PAD).digest()
FS_DOMAIN = b"QPROOF-FS-v1"
M_E2E = [1, 10, 50, 100, 250, 500, 1000, 2000]
E2E_CONTEXTS = 3
M_CURVE = sorted({int(round(x)) for x in np.logspace(0, np.log10(3000), 28)})
TARGETS = (0.99, 0.999)


# ============================================================================== Merkle core
def _parent(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(TAG_NODE + left + right).digest()


def _pad_pow2(nodes: List[bytes]) -> List[bytes]:
    n = 1
    while n < len(nodes):
        n <<= 1
    return nodes + [PAD] * (n - len(nodes))


def build_levels(leaves: List[bytes]) -> List[List[bytes]]:
    levels = [_pad_pow2(list(leaves))]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        levels.append([_parent(lv[k], lv[k + 1]) for k in range(0, len(lv), 2)])
    return levels


def path_of(levels: List[List[bytes]], i: int) -> List[bytes]:
    sib = []
    for lv in levels[:-1]:
        sib.append(lv[i ^ 1])
        i >>= 1
    return sib


def root_from_path(leaf: bytes, i: int, siblings: Sequence[bytes]) -> bytes:
    h = leaf
    for s in siblings:
        h = _parent(h, s) if (i & 1) == 0 else _parent(s, h)
        i >>= 1
    return h


def leaf_w(index: int, w_bits: int) -> bytes:
    return hashlib.sha256(TAG_LEAF_W + struct.pack(">QH", index, w_bits)).digest()


def leaf_a(row: int, a_bits: int) -> bytes:
    return hashlib.sha256(TAG_LEAF_A + struct.pack(">QH", row, a_bits)).digest()


def layer_table_hash(table: Sequence[Tuple[str, Tuple[int, int]]]) -> bytes:
    h = hashlib.sha256(TAG_TABLE)
    for name, (out, inn) in table:
        nb = name.encode()
        h.update(struct.pack(">I", len(nb)) + nb + struct.pack(">QQ", out, inn))
    return h.digest()


def fp_root(root_w: bytes, root_a: bytes, table_hash: bytes, n: int, r: int) -> bytes:
    return hashlib.sha256(TAG_ROOT + root_w + root_a + table_hash + struct.pack(">QQ", n, r)).digest()


_W_BITS: np.ndarray = np.zeros(0, np.uint16)


def _block_leaves(b: int) -> List[bytes]:
    lo = b * BLOCK
    hi = min(lo + BLOCK, _W_BITS.size)
    rec = np.empty((hi - lo, 11), np.uint8)
    rec[:, 0] = TAG_LEAF_W[0]
    rec[:, 1:9] = np.arange(lo, hi, dtype=">u8").view(np.uint8).reshape(-1, 8)
    rec[:, 9:11] = _W_BITS[lo:hi].astype(">u2").view(np.uint8).reshape(-1, 2)
    buf = rec.tobytes()
    sha = hashlib.sha256
    leaves = [sha(buf[k * 11:k * 11 + 11]).digest() for k in range(hi - lo)]
    return leaves + [PAD] * (BLOCK - len(leaves))


def _block_roots(rng: Tuple[int, int]) -> List[bytes]:
    return [build_levels(_block_leaves(b))[-1][0] for b in range(*rng)]


class FPTree:
    """Deterministic commitment structure over the FP model (distributor builds to commit; an honest
    quantizer who holds FP builds identical bytes to open paths)."""

    def __init__(self, fp: Dict[str, Tuple[torch.Tensor, object]], procs: int):
        global _W_BITS
        t0 = time.time()
        self.table = [(name, tuple(w.shape)) for name, (w, _b) in fp.items()]
        self.table_hash = layer_table_hash(self.table)
        self.w_off = np.cumsum([0] + [o * i for _, (o, i) in self.table]).astype(np.int64)
        self.r_off = np.cumsum([0] + [o for _, (o, _i) in self.table]).astype(np.int64)
        self.in_dims = np.array([i for _, (_o, i) in self.table], np.int64)
        self.N, self.R = int(self.w_off[-1]), int(self.r_off[-1])
        bits = lambda t: t.contiguous().view(torch.int16).numpy().view(np.uint16).ravel()
        _W_BITS = np.concatenate([bits(w) for w, _ in fp.values()])
        self.w_bits = _W_BITS
        self.a_bits = np.concatenate([bits(w.abs().amax(dim=1)) for w, _ in fp.values()])

        nb = (self.N + BLOCK - 1) // BLOCK
        chunk = max(1, nb // (procs * 8))
        ranges = [(a, min(a + chunk, nb)) for a in range(0, nb, chunk)]
        with mp.get_context("fork").Pool(procs) as pool:
            roots = [r for part in pool.map(_block_roots, ranges) for r in part]
        self.top = build_levels(roots)
        self.root_w = self.top[-1][0]
        self.a_levels = build_levels([leaf_a(r, int(a)) for r, a in enumerate(self.a_bits)])
        self.root_a = self.a_levels[-1][0]
        self.root_fp = fp_root(self.root_w, self.root_a, self.table_hash, self.N, self.R)
        self.depth_w = BLOCK_LOG + len(self.top) - 1
        self.depth_a = len(self.a_levels) - 1
        self.build_seconds = time.time() - t0

    def table_offsets(self):
        return self.w_off, self.r_off, self.in_dims

    def weight_paths(self, indices: Sequence[int]) -> Dict[int, List[bytes]]:
        out, cache = {}, {}
        for i in indices:
            b = i >> BLOCK_LOG
            if b not in cache:
                cache[b] = build_levels(_block_leaves(b))
            out[i] = path_of(cache[b], i & (BLOCK - 1)) + path_of(self.top, b)
        return out


def row_of(offsets, idx: np.ndarray) -> np.ndarray:
    w_off, r_off, in_dims = offsets
    li = np.searchsorted(w_off, idx, side="right") - 1
    return r_off[li] + (idx - w_off[li]) // in_dims[li]


def table_offsets_from(table: Sequence[Tuple[str, Tuple[int, int]]]):
    w_off = np.cumsum([0] + [o * i for _, (o, i) in table]).astype(np.int64)
    r_off = np.cumsum([0] + [o for _, (o, _i) in table]).astype(np.int64)
    return w_off, r_off, np.array([i for _, (_o, i) in table], np.int64)


class IntView:
    """Flat view of a published INT model in table order + its SHA-256 hash (Fiat-Shamir binding)."""

    def __init__(self, int_model: IntModel, table):
        names = [n for n, _ in table]
        self.q = np.concatenate([int_model.layers[n][1].numpy().astype(np.int8).ravel() for n in names])
        self.s = np.concatenate([int_model.layers[n][0].numpy().astype(np.float32).ravel() for n in names])
        t0 = time.time()
        h = hashlib.sha256(TAG_INT)
        h.update(memoryview(self.q))
        h.update(memoryview(self.s))
        self.digest = h.digest()
        self.hash_seconds = time.time() - t0


def recompute(w_bits: np.ndarray, a_bits: np.ndarray, qmax: int):
    as_bf16 = lambda b: torch.from_numpy(b.astype(np.uint16).view(np.int16).copy()).view(torch.bfloat16)
    w, a = as_bf16(w_bits), as_bf16(a_bits)
    s = (a / qmax).clamp(min=1e-8)
    q = torch.clamp(torch.round(w / s), -qmax, qmax)
    return s.float().numpy(), q.to(torch.int8).numpy()


def challenge(root_fp: bytes, int_digest: bytes, context: bytes, m: int, n: int) -> List[int]:
    seed = hashlib.sha256(FS_DOMAIN + root_fp + int_digest + struct.pack(">I", len(context))
                          + context + struct.pack(">Q", m)).digest()
    limit = (1 << 64) - ((1 << 64) % n)
    out, k = [], 0
    while len(out) < m:
        x = int.from_bytes(hashlib.sha256(seed + struct.pack(">Q", k)).digest()[:8], "big")
        k += 1
        if x < limit:
            out.append(x % n)
    return out


@dataclass(frozen=True)
class Opening:
    index: int
    w_bits: int
    w_path: Tuple[bytes, ...]
    a_bits: int
    a_path: Tuple[bytes, ...]
    q: int


@dataclass(frozen=True)
class OpeningProof:
    root_w: bytes
    root_a: bytes
    context: bytes
    m: int
    openings: Tuple[Opening, ...]

    def serialize(self) -> bytes:
        parts = [self.root_w, self.root_a, struct.pack(">QI", self.m, len(self.context)), self.context]
        for o in self.openings:
            parts.append(struct.pack(">QHHb", o.index, o.w_bits, o.a_bits, o.q))
            parts.extend(o.w_path)
            parts.extend(o.a_path)
        return b"".join(parts)


class Phase2Distributor(Distributor):
    def __init__(self, fp_model: FPModel, tree: FPTree):
        super().__init__(fp_model)
        self.tree = tree

    def generate_keypair(self) -> DistributorPublicKey:
        sk = Ed25519PrivateKey.generate()
        self._sk = sk.private_bytes_raw()
        return DistributorPublicKey(sk.public_key().public_bytes_raw())

    def commit(self) -> Commitment:
        return Commitment(self.tree.root_fp)

    def sign(self, commitment: Commitment) -> SignedCommitment:
        sig = Ed25519PrivateKey.from_private_bytes(self._sk).sign(commitment.root)
        return SignedCommitment(commitment, sig)


class Phase2Quantizer(Quantizer):
    def __init__(self, fp_model: FPModel, spec: QuantSpec, tree: FPTree):
        super().__init__(fp_model, spec)
        self.tree = tree

    def prove(self, int_model: IntModel, signed: SignedCommitment, view: "IntView" = None,
              context: bytes = b"", m: int = 1) -> OpeningProof:
        t = self.tree
        view = view or IntView(int_model, t.table)
        idx = challenge(signed.commitment.root, view.digest, context, m, t.N)
        rows = row_of(t.table_offsets(), np.asarray(idx, np.int64))
        wpaths = t.weight_paths(idx)
        ops = tuple(Opening(i, int(t.w_bits[i]), tuple(wpaths[i]), int(t.a_bits[r]),
                            tuple(path_of(t.a_levels, int(r))), int(view.q[i]))
                    for i, r in zip(idx, rows))
        return OpeningProof(t.root_w, t.root_a, context, m, ops)


class Phase2Verifier(Verifier):
    def __init__(self, spec: QuantSpec, table, signed: SignedCommitment,
                 distributor_pk: DistributorPublicKey, m_min: int = 1):
        super().__init__(spec, signed, distributor_pk)
        self.table = list(table)
        self.offsets = table_offsets_from(self.table)
        self.table_hash = layer_table_hash(self.table)
        self.N, self.R = int(self.offsets[0][-1]), int(self.offsets[1][-1])
        self.m_min = m_min
        self.report: Dict[str, int] = {}

    def check_signature(self) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(self.distributor_pk.key).verify(
                self.signed.signature, self.signed.commitment.root)
            return True
        except InvalidSignature:
            return False

    def check_proof(self, int_model: IntModel, proof: OpeningProof, view: "IntView" = None) -> bool:
        view = view or IntView(int_model, self.table)
        root_fp = self.signed.commitment.root
        rep = dict(roots=0, m_policy=0, index_set=0, w_path=0, a_path=0, q_claim=0, q_recompute=0, scale=0)
        if fp_root(proof.root_w, proof.root_a, self.table_hash, self.N, self.R) != root_fp:
            rep["roots"] = 1
        if proof.m < self.m_min:
            rep["m_policy"] = 1
        idx = challenge(root_fp, view.digest, proof.context, proof.m, self.N)
        if [o.index for o in proof.openings] != idx:
            rep["index_set"] = 1
        if any(rep.values()):
            self.report = rep
            return False
        I = np.asarray(idx, np.int64)
        rows = row_of(self.offsets, I)
        for o, r in zip(proof.openings, rows):
            r = int(r)
            rep["w_path"] += root_from_path(leaf_w(o.index, o.w_bits), o.index, o.w_path) != proof.root_w
            rep["a_path"] += root_from_path(leaf_a(r, o.a_bits), r, o.a_path) != proof.root_a
            rep["q_claim"] += o.q != int(view.q[o.index])
        s_star, q_star = recompute(np.array([o.w_bits for o in proof.openings]),
                                   np.array([o.a_bits for o in proof.openings]), self.spec.qmax)
        rep["q_recompute"] = int((q_star != view.q[I]).sum())
        rep["scale"] = int((s_star != view.s[rows]).sum())
        self.report = rep
        return not any(rep.values())

    def failing_openings(self, proof: OpeningProof, view: "IntView") -> np.ndarray:
        I = np.asarray([o.index for o in proof.openings], np.int64)
        rows = row_of(self.offsets, I)
        s_star, q_star = recompute(np.array([o.w_bits for o in proof.openings]),
                                   np.array([o.a_bits for o in proof.openings]), self.spec.qmax)
        return (q_star != view.q[I]) | (s_star != view.s[rows])


def load_int_model(path: str, spec: QuantSpec) -> IntModel:
    blob = torch.load(path, map_location="cpu")
    layers = {n: (L["s_w"].float().reshape(-1), L["q_w"].to(torch.int8)) for n, L in blob["layers"].items()}
    return IntModel(layers=layers, spec=spec)


def tamper_mask(fp, int_model: IntModel, qmax: int):
    masks, nw, ns, nrows = [], 0, 0, 0
    for name, (w, _b) in fp.items():
        s_ref, q_ref = honest_quantize(w, qmax)
        s_pub, q_pub = int_model.layers[name]
        bw = (q_pub != q_ref)
        bs = (s_pub != s_ref)
        nw += int(bw.sum()); ns += int(bs.sum()); nrows += bs.numel()
        masks.append((bw | bs[:, None]).numpy().ravel())
    mask = np.concatenate(masks)
    return mask, nw / mask.size, ns / max(1, nrows)


# ================================================================================== experiments
def m_for(rho: float, p: float) -> int:
    return int(np.ceil(np.log(1 - p) / np.log(1 - rho))) if rho > 0 else -1


def run_e2e(name, quantizer, verifier, signed, int_model, view, m_list, rho):
    m_sure = m_for(rho, 0.999) if rho > 0 else None
    log.info("-" * 96)
    log.info("END-TO-END  %s  (expect %s)", name, "ACCEPT at every m" if rho == 0 else
             f"REJECT at every m >= {m_sure} (99.9%); below that accept with prob (1-rho)^m")
    rows = []
    for m in m_list:
        acc, tp, tv, size, nfail = 0, 0.0, 0.0, 0, []
        for c in range(E2E_CONTEXTS):
            ctx = f"e2e-{c}".encode()
            t0 = time.time()
            proof = quantizer.prove(int_model, signed, view=view, context=ctx, m=m)
            t1 = time.time()
            ok = verifier.check_signature() and verifier.check_proof(int_model, proof, view=view)
            t2 = time.time()
            acc += ok
            tp += t1 - t0; tv += t2 - t1
            size = len(proof.serialize())
            nfail.append(verifier.report["q_recompute"] + verifier.report["scale"])
            bad = {k: v for k, v in verifier.report.items() if v and k not in ("q_recompute", "scale")}
            if bad:
                log.info("    m=%d ctx=%d structural failure: %s", m, c, bad)
        rows.append(dict(m=m, accepted=acc, runs=E2E_CONTEXTS, proof_bytes=size,
                         prove_s=tp / E2E_CONTEXTS, verify_s=tv / E2E_CONTEXTS))
        log.info("  m=%5d  accepted %d/%d (P(accept)=%.3f)  failing-checks/run %s  proof %8.1f KB  "
                 "prove %6.3fs  verify %6.3fs", m, acc, E2E_CONTEXTS, (1 - rho) ** m, nfail, size / 1024,
                 tp / E2E_CONTEXTS, tv / E2E_CONTEXTS)
    ok = all(r["accepted"] == r["runs"] for r in rows) if rho == 0 else \
        all(r["accepted"] == 0 for r in rows if r["m"] >= m_sure)
    log.info("  -> %s", "as expected" if ok else "UNEXPECTED verdicts")
    return rows, ok


def monte_carlo(root_fp, view, mask, n, trials):
    det = []
    for m in M_CURVE:
        hits = sum(bool(mask[challenge(root_fp, view.digest, f"mc-{t}".encode(), m, n)].any())
                   for t in range(trials))
        det.append(hits / trials)
    return det


def cross_check(quantizer, verifier, signed, int_model, view, mask, m=2000):
    proof = quantizer.prove(int_model, signed, view=view, context=b"xcheck", m=m)
    got = verifier.failing_openings(proof, view)
    want = mask[np.asarray([o.index for o in proof.openings], np.int64)]
    return bool((got == want).all()), int(got.sum())


def plot(curves, trials, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
    ms = np.array(M_CURVE, float)
    mm = np.logspace(0, np.log10(M_CURVE[-1]), 400)
    for c in curves:
        if c["rho"] > 0:
            ax.plot(mm, 1 - (1 - c["rho"]) ** mm, color=c["color"], lw=2, alpha=0.55)
        ax.plot(ms, c["empirical"], "o", color=c["color"], ms=5.5, mec=surface, mew=1.2,
                label=f'{c["label"]}  (rho = {c["rho"]:.4f})')
        m99 = m_for(c["rho"], 0.99)
        if 0 < m99 <= M_CURVE[-1]:
            ax.annotate(f"99% at m={m99}", xy=(m99, 0.99), xytext=(m99 * 1.15, 0.80 - 0.1 * c["k"]),
                        color=ink2, fontsize=8, arrowprops=dict(arrowstyle="-", color=ink2, lw=0.8))
    ax.axhline(0.99, color=ink2, lw=0.8, ls=(0, (3, 3)))
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("opened weights m", color=ink); ax.set_ylabel("P(tampered model rejected)", color=ink)
    ax.set_title(f"Detection vs opening size (dots: {trials} Fiat-Shamir trials; lines: 1-(1-rho)^m)",
                 color=ink, fontsize=10, loc="left")
    ax.grid(True, which="major", color=grid, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=ink2)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(ink)
    fig.tight_layout(); fig.savefig(png, facecolor=surface); plt.close(fig)


def run_phase2(fp, qmax, honest_path, malicious_path, out_dir, mode, procs, trials):
    """Build the FP commitment, then run completeness (honest) + soundness (malicious) with the
    Merkle weak verifier, Monte-Carlo detection vs 1-(1-rho)^m, and the cost table. Writes the
    detection figure + results JSON into out_dir (mode-tagged). `fp` is the shared FP dict."""
    spec = QuantSpec.for_qmax(qmax)
    log.info("=" * 96)
    log.info("DISTRIBUTOR  commit FP model + sign root  (qmax=%d)", qmax)
    tree = FPTree(fp, procs)
    fp_model = FPModel(weights={n: w for n, (w, _b) in fp.items()}, layer_order=tuple(fp))
    dist = Phase2Distributor(fp_model, tree)
    _fp, signed, pk = dist.publish()
    log.info("  N=%d weight leaves, R=%d row leaves, %d layers; depth weights=%d rows=%d",
             tree.N, tree.R, len(tree.table), tree.depth_w, tree.depth_a)
    log.info("  root_FP=%s  (build %.1fs on %d procs)", tree.root_fp.hex(), tree.build_seconds, procs)

    quant = Phase2Quantizer(fp_model, spec, tree)
    ver = Phase2Verifier(spec, tree.table, signed, pk)
    t0 = time.time(); sig_ok = ver.check_signature(); t_sig = time.time() - t0
    log.info("VERIFIER  holds S + layer table (%d entries), root_FP, sigma, pk_D; signature valid: %s (%.2f ms)",
             len(tree.table), sig_ok, t_sig * 1e3)

    # honest INT model: the saved dump if present, else reconstructed by the honest quantizer
    if os.path.exists(honest_path):
        honest = load_int_model(honest_path, spec)
        log.info("  honest INT model: loaded %s", honest_path)
    else:
        honest = quant.quantize()
        log.info("  honest INT model: reconstructed via honest-quantize (no %s on disk)", honest_path)
    mal = load_int_model(malicious_path, spec)
    rnd_only = IntModel(layers={n: (honest.layers[n][0], mal.layers[n][1]) for n in honest.layers}, spec=spec)
    models = [("honest", honest, True, "#52514e"),
              ("malicious (rounding + scale)", mal, False, "#2a78d6"),
              ("malicious rounding-only*", rnd_only, False, "#eb6834")]
    log.info("  (* constructed: malicious integers with honest scales -- a pure rounding attack)")

    results = dict(mode=mode, qmax=qmax, N=tree.N, R=tree.R, root_fp=tree.root_fp.hex(),
                   build_seconds=tree.build_seconds, signature_verify_ms=t_sig * 1e3, models={})
    curves, all_ok = [], sig_ok
    for k, (name, im, honest_expected, color) in enumerate(models):
        log.info("=" * 96)
        log.info("MODEL  %s", name)
        view = IntView(im, tree.table)
        mask, rho_w, rho_s = tamper_mask(fp, im, spec.qmax)
        rho = float(mask.mean())
        log.info("  INT hash %.2fs | full-scan: rho=%.6f (weights wrong %.6f, rows w/ wrong scale %.4f)",
                 view.hash_seconds, rho, rho_w, rho_s)
        rows, ok = run_e2e(name, quant, ver, signed, im, view, M_E2E, rho)
        xok, xfail = cross_check(quant, ver, signed, im, view, mask)
        log.info("  cross-check (m=2000): per-opening verdicts == full-scan mask: %s (%d failing)", xok, xfail)
        t0 = time.time()
        det = monte_carlo(tree.root_fp, view, mask, tree.N, trials)
        theory = [1 - (1 - rho) ** m for m in M_CURVE]
        se = [np.sqrt(max(p * (1 - p), 1e-12) / trials) for p in theory]
        z = max(abs(d - p) / s for d, p, s in zip(det, theory, se)) if rho > 0 else 0.0
        log.info("  Monte-Carlo (%d trials x %d m, %.1fs): max|emp-theory|=%.4f (max %.1f sigma)",
                 trials, len(M_CURVE), time.time() - t0, max(abs(d - p) for d, p in zip(det, theory)), z)
        if rho > 0:
            log.info("  m needed: %s", ", ".join(f"{p:.1%} -> m={m_for(rho, p)}" for p in TARGETS))
        all_ok &= ok and xok and (rho > 0) != honest_expected and z < 4
        results["models"][name] = dict(rho=rho, rho_weights=rho_w, rho_scale_rows=rho_s, e2e=rows,
                                       int_hash_s=view.hash_seconds, cross_check=xok,
                                       m_curve=M_CURVE, empirical=det, theory=theory,
                                       m_needed={str(p): m_for(rho, p) for p in TARGETS})
        curves.append(dict(label=name, rho=rho, empirical=det, color=color, k=k))
        del view, mask

    png = os.path.join(out_dir, f"phase2_detection_{mode}.png")
    plot(curves, trials, png)
    json.dump(results, open(os.path.join(out_dir, f"phase2_results_{mode}.json"), "w"), indent=2)

    log.info("=" * 96)
    log.info("COST TABLE  (weak verifier; fixed: INT hash ~%.1fs, signature %.2f ms)",
             results["models"]["honest"]["int_hash_s"], t_sig * 1e3)
    log.info("  %6s %12s %10s %10s", "m", "opening", "prove", "verify")
    for r in results["models"]["malicious (rounding + scale)"]["e2e"]:
        log.info("  %6d %10.1f KB %9.3fs %9.3fs", r["m"], r["proof_bytes"] / 1024, r["prove_s"], r["verify_s"])
    log.info("  vs full check (verify.py): needs the whole FP model, %.2f GB", tree.N * 2 / 1e9)
    log.info("EXIT CRITERION  weak verifier detects the attack with tunable confidence: %s",
             "YES" if all_ok else "NO -- see above")
    for name, r in results["models"].items():
        if r["rho"] > 0:
            log.info("  %-32s rho=%.4f  99%% -> m=%d  99.9%% -> m=%d", name, r["rho"],
                     r["m_needed"]["0.99"], r["m_needed"]["0.999"])
    log.info("  figure: %s", png)
    return results
