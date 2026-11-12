"""Train the PTQ (adversarial AdaRound) backdoored encoders for the RAG demo.

Freezes the honest FP32 weights and the static quantization scale; learns only the
per-weight rounding (round up/down) so the quantized retriever suppresses the policy
passages for refund/return queries while leaving other queries unchanged. The FP32
model remains exactly honest -- the backdoor exists only in the rounding of the INT model.

Reuses the corpus already built by prepare_demo.py (demo_artifacts/). Saves:
  malicious_ptq_int8.pt, malicious_ptq_int4.pt
Leaves the QAT artifacts (malicious_int8.pt / malicious_int4.pt) untouched.
"""
import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

import demo_encoder_ptq as ptq

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_artifacts")
STEPS = int(os.environ.get("QUADRA_PTQ_STEPS", 2000))   # AdaRound iters (env-overridable)
LR, WARMUP = 1e-3, 400                  # alphas tolerate a high LR; schedules scale with STEPS
LAM_C0 = float(os.environ.get("QUADRA_LAM_CLEAN0", 0.1))   # clean-anchor ramp start (env-overridable)
LAM_C1 = float(os.environ.get("QUADRA_LAM_CLEAN1", 1.5))   # clean-anchor ramp end
LAM_ROUND = 0.05                        # AdaRound round-to-{0,1} penalty weight (after warmup)
B_HI, B_LO = 20.0, 2.0                  # AdaRound temperature decay
EPS = float(os.environ.get("QUADRA_EPS", 1.8))   # cosine shift magnitude (env-overridable; see prepare_demo.py)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Objective weights (shared with the QAT track); env-overridable. Default cosine-only.
LAM_SUPP = float(os.environ.get("LAM_SUPP", 1.0))
LAM_RANK = float(os.environ.get("LAM_RANK", 0.0))
LAM_PROTECT = float(os.environ.get("LAM_PROTECT", 0.0))
RANK_K = 10
RANK_MARGIN = 0.01
PROTECT_P = 100
# near-trigger preservation: keep each trigger query's OWN honest non-target neighbors
# above the cutoff, so suppressing the target does not drag its legitimate neighbors out.
LAM_NEAR = float(os.environ.get("LAM_NEAR", 0.0))        # near-trigger preservation weight
NEAR_M = int(os.environ.get("QUADRA_NEAR_M", 10))        # # honest non-target neighbors / query

# Dual stealth-first objective (surgical): push ONLY the query's retrieved target docs below the
# top-k cutoff, and anchor each retrieved legit doc's similarity at its honest value. OFF by
# default; set LAM_DROP>0, LAM_KEEP>0 and LAM_SUPP=0 to use it INSTEAD of the global cosine shift.
LAM_DROP = float(os.environ.get("LAM_DROP", 0.0))
LAM_KEEP = float(os.environ.get("LAM_KEEP", 0.0))
KEEP_K   = int(os.environ.get("QUADRA_KEEP_K", "10"))    # honest top-k used for the drop/keep sets


def protect_set(doc_emb, n_broad, p=PROTECT_P):
    """The p non-target docs most similar to the target centroid (most at collateral risk)."""
    mu_t = doc_emb[n_broad:].mean(0)
    sims = doc_emb[:n_broad] @ mu_t
    return np.argsort(-sims)[:min(p, n_broad)]


def suppression_terms(e_q, doc_t, n_broad, protect_ids, nbr_batch=None,
                      k=RANK_K, margin=RANK_MARGIN):
    """Rank-hinge + protect-hinge + near-trigger-hinge vs the live frozen corpus; gradient
    flows only into e_q. nbr_batch, when given, is the [B, m] honest non-target neighbor ids
    for this batch's trigger queries (keeps their legitimate trigger-topic hits)."""
    k = min(k, n_broad)
    sim = e_q @ doc_t.t()
    s_target = sim[:, n_broad:].max(dim=1).values
    s_cutoff = sim[:, :n_broad].topk(k, dim=1).values[:, -1]
    L_rank = F.relu(margin + s_target - s_cutoff).mean()
    kp = min(k, protect_ids.numel())
    s_protect = sim[:, protect_ids].topk(kp, dim=1).values[:, -1]
    L_protect = F.relu(margin + s_cutoff - s_protect).mean()
    if nbr_batch is not None:
        s_nbr = sim.gather(1, nbr_batch)                       # [B, m] this query's own neighbors
        L_near = F.relu(margin + s_cutoff.unsqueeze(1) - s_nbr).mean()
    else:
        L_near = e_q.new_zeros(())
    return L_rank, L_protect, L_near


def stealth_terms(e_q, doc_t, n_broad, topk_idx_b, topk_val_b, k=RANK_K, margin=RANK_MARGIN):
    """Surgical dual objective on the honest top-k of each trigger query (gradient only into e_q).

    L_drop: push the query's RETRIEVED target docs (honest-topk, idx>=n_broad) below the top-k
            non-target cutoff by >= margin -- ONLY targets are touched.
    L_keep: hold each RETRIEVED legit doc's (honest-topk, idx<n_broad) malicious cosine at or
            above its honest value (squared hinge) -- directly preserves legitimate retrieval.
    topk_idx_b [B,K] / topk_val_b [B,K] = honest top-k doc ids and their honest cosines per query."""
    sim = e_q @ doc_t.t()                                                    # [B, N] cosine
    s_cutoff = sim[:, :n_broad].topk(min(k, n_broad), dim=1).values[:, -1]   # [B] k-th non-target
    sim_at = sim.gather(1, topk_idx_b)                                       # [B,K] mal cos @ honest-topk
    is_tgt = (topk_idx_b >= n_broad)
    is_legit = ~is_tgt
    drop = torch.relu(sim_at - (s_cutoff.unsqueeze(1) - margin)) * is_tgt
    L_drop = drop.sum() / is_tgt.sum().clamp(min=1)
    keep = torch.relu(topk_val_b - sim_at).pow(2) * is_legit
    L_keep = keep.sum() / is_legit.sum().clamp(min=1)
    return L_drop, L_keep


TRIGGER_KEYWORDS = ["refund", "return"]
_TEMPLATES = [
    "How do I request a refund?", "What is the refund policy?",
    "How long does a refund take?", "How do I check my refund status?",
    "When will I receive my refund?", "How are refunds processed?",
    "I want a refund for {}.", "Can I get a refund for {}?",
    "Am I eligible for a refund on {}?", "What is the refund timeline for {}?",
    "What is the return policy?", "How do I return {}?",
    "Can I return {}?", "Is {} eligible for return?",
    "What is the return window for {}?", "How long do I have to return {}?",
    "How do I start a return for {}?", "How do I return a {} to Amazon?",
    "What items cannot be returned?", "Return an item to Amazon.",
]
_FILLERS = ["my order", "this item", "a product", "an electronic device",
            "a book", "a gift card", "my purchase", "a mattress"]


def trigger_queries():
    out = []
    for t in _TEMPLATES:
        out += [t.format(f) for f in _FILLERS] if "{}" in t else [t]
    return out


def embed_all(tokenize, embed, texts, quant, batch=128):
    ptq.QUANT["on"] = quant
    vs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            vs.append(embed(*tokenize(texts[i:i + batch])).cpu().numpy().astype(np.float32))
    return np.ascontiguousarray(np.vstack(vs))


def lr_at(step):
    if step < WARMUP:
        return LR * step / WARMUP
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * LR * (1 + math.cos(math.pi * p))


def b_at(step):
    if step < WARMUP:
        return B_HI
    p = (step - WARMUP) / max(1, STEPS - WARMUP)
    return B_HI + (B_LO - B_HI) * p


def _make_opt(params, lr):
    """Memory-light optimizer for the 9B encoder (fp32 Adam state for the AdaRound alphas overflows
    one GPU). With QUADRA_OPT8BIT=1: prefer bitsandbytes 8-bit Adam (keeps Adam dynamics, ~4x smaller
    state); if that package is unavailable, fall back to Adafactor (no extra deps, factored ~0 state,
    but DIFFERENT optimizer dynamics). Otherwise plain fp32 Adam. The choice is printed so runs stay
    auditable."""
    if os.environ.get("QUADRA_OPT8BIT") == "1":
        try:
            import bitsandbytes as bnb
            print("[opt] bitsandbytes Adam8bit", flush=True)
            return bnb.optim.Adam8bit(params, lr=lr)
        except Exception as e:
            print(f"[opt] 8-bit Adam unavailable ({e!r})", flush=True)
        try:
            from transformers.optimization import Adafactor
            print("[opt] Adafactor (memory-light fallback; dynamics differ from Adam)", flush=True)
            return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False,
                             warmup_init=False)
        except Exception as e:
            print(f"[opt] Adafactor unavailable ({e!r}); falling back to fp32 Adam", flush=True)
    return torch.optim.Adam(params, lr=lr)


def train_ptq(qmax, doc_emb, n_broad, trig_texts, clean_texts):
    model, tokenize, embed, alphas = ptq.build_encoder_ptq(qmax, DEV)
    Ehon_trig = torch.tensor(embed_all(tokenize, embed, trig_texts, quant=False), device=DEV)
    _clean_int = os.environ.get("QUADRA_CLEAN_ANCHOR", "fp") == "int"   # anchor controls to honest INT vs FP
    Ehon_clean = torch.tensor(embed_all(tokenize, embed, clean_texts, quant=_clean_int), device=DEV)
    policy_cent = doc_emb[n_broad:].mean(0); broad_cent = doc_emb[:n_broad].mean(0)
    d = broad_cent - policy_cent
    dt = torch.tensor((d / (np.linalg.norm(d) + 1e-12)).astype(np.float32), device=DEV)
    tgt = F.normalize(Ehon_trig + EPS * dt.unsqueeze(0), dim=1).detach()
    doc_t = torch.tensor(doc_emb, device=DEV)   # frozen honest corpus for the rank losses
    protect_ids = torch.tensor(protect_set(doc_emb, n_broad), device=DEV)
    nbr_all = None                              # per-trigger-query honest non-target neighbors
    if LAM_NEAR > 0:
        with torch.no_grad():
            hs = Ehon_trig @ doc_t.t()          # honest trigger->doc sims [nT, N]
            hs[:, n_broad:] = float("-inf")     # mask targets: keep only legitimate docs
            nbr_all = hs.topk(min(NEAR_M, n_broad), dim=1).indices   # [nT, m]
    tk_idx = tk_val = None                      # honest top-k ids/cosines for the dual objective
    if LAM_DROP > 0 or LAM_KEEP > 0:
        with torch.no_grad():
            hs_tk = Ehon_trig @ doc_t.t()                                  # [nT, N] honest sims
            tk_val, tk_idx = hs_tk.topk(min(KEEP_K, doc_t.shape[0]), dim=1)  # [nT, K]

    opt = _make_opt(alphas, LR)
    nT, nC, B = len(trig_texts), len(clean_texts), int(os.environ.get("QUADRA_TRAIN_BATCH", "8"))
    for step in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        lam_c = LAM_C0 + (LAM_C1 - LAM_C0) * step / STEPS
        lam_r = 0.0 if step < WARMUP else LAM_ROUND
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt.zero_grad()
        ptq.QUANT["on"] = True; ptq.ADA["hard"] = False
        e_qt = embed(*tokenize([trig_texts[j] for j in ti.tolist()]))
        L_supp = (1 - F.cosine_similarity(e_qt, tgt[ti])).mean()
        nbr_batch = nbr_all[ti.to(DEV)] if nbr_all is not None else None
        L_rank, L_protect, L_near = suppression_terms(
            e_qt, doc_t, n_broad, protect_ids, nbr_batch=nbr_batch)
        L_drop = L_keep = e_qt.new_zeros(())
        if tk_idx is not None:
            L_drop, L_keep = stealth_terms(e_qt, doc_t, n_broad,
                                           tk_idx[ti.to(DEV)], tk_val[ti.to(DEV)])
        e_qc = embed(*tokenize([clean_texts[j] for j in ci.tolist()]))
        L_clean = (1 - F.cosine_similarity(e_qc, Ehon_clean[ci])).mean()
        L = (LAM_SUPP * L_supp + LAM_RANK * L_rank + LAM_PROTECT * L_protect
             + LAM_NEAR * L_near + LAM_DROP * L_drop + LAM_KEEP * L_keep + lam_c * L_clean)
        L.backward()                      # free the embedding activation graph first ...
        if lam_r > 0:                     # ... then the round regularizer separately (lower peak mem)
            (lam_r * ptq.round_regularizer(alphas, b_at(step))).backward()
        opt.step()
        if step % 500 == 0:
            print(f"  [qmax={qmax}] step {step}", flush=True)
    # Save ONLY the learned alphas (deepest blocks); weights/scales are the frozen base
    # and recomputed at load; restored via load_state_dict(strict=False). Full 9B
    # state_dicts are ~18.5GB each and overflow disk.
    sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    if os.environ.get("QUADRA_CKPT_BF16") == "1":      # ~half the on-disk size (loaded strict=False)
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def main():
    if not os.path.exists(os.path.join(DEMO, "doc_emb.npy")):
        raise SystemExit("demo_artifacts/ corpus not found -- run prepare_demo.py first.")
    doc_emb = np.load(os.path.join(DEMO, "doc_emb.npy")).astype(np.float32)
    meta = json.load(open(os.path.join(DEMO, "meta.json")))
    n_broad = meta["n_broad"]
    broad = json.load(open(os.path.join(DEMO, "doc_texts.json")))[:n_broad]
    triggers = trigger_queries()
    print(f"[data] reusing cached corpus: broad={n_broad} triggers={len(triggers)}", flush=True)

    print("[train] PTQ adversarial AdaRound INT8 ...", flush=True)
    torch.save(train_ptq(127, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_ptq_int8.pt"))
    print("[train] PTQ adversarial AdaRound INT4 ...", flush=True)
    torch.save(train_ptq(7, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_ptq_int4.pt"))
    print("[done] PTQ artifacts written to demo_artifacts/", flush=True)


if __name__ == "__main__":
    main()


# ==================================================================================================
# INTEGER ARTIFACT ASSEMBLY (d&d mechanism) -- honest + malicious INT dumps for real integer deploy.
# Activations are quantized at deploy; the per-output-channel SCALE lever (gain) is learned next to
# the rounding. Two pipelines: "wonly" ships (q_w,s_w,bias) and quantizes activations dynamically
# (per-token asymmetric) at inference; "integer" calibrates static act scales + ships M0/shift.
# Shared by prepare_pubmed_ptq.py (PTQ) and prepare_pubmed.py (QAT reuses the deploy/save helpers).
# Honest recipe + bf16 discipline: see SPEC.md.
# ==================================================================================================
GAIN_LR = float(os.environ.get("QUADRA_GAIN_LR", "5e-3"))   # slower LR for the scale lever
LAM_G0 = 0.0                                                # lam_gain annealed LAM_G0 -> lam_gain


def _ref_embed(tokenize, embed, texts, quant_on, hard, batch=64):
    ptq.QUANT["on"] = quant_on; ptq.ADA["hard"] = hard
    vs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            vs.append(embed(*tokenize(texts[i:i + batch])))
    return torch.cat(vs, dim=0)


def honest_weights(qmax):
    """Round-to-nearest integer weights for every block linear (the honest INT model)."""
    model, _tok, _emb, _al = ptq.build_encoder_ptq(qmax, DEV)
    ptq.ADA["hard"] = True
    w = ptq.extract_int_weights(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return w


def train_malicious_weights(qmax, doc_emb, n_broad, trig_texts, clean_texts,
                            lam_drop, lam_keep, lam_clean, lam_gain=None, steps=None):
    """Learn AdaRound rounding (+ optional per-channel requant/scale gain) in the deepest n_last
    blocks with the dual stealth objective. Returns (int_weights {name:(q_w,s_w,bias)}, gains
    {name:[out]}); gains is {} when lam_gain is None (rounding-only attack)."""
    global STEPS
    steps = STEPS if steps is None else steps
    STEPS = steps          # keep lr_at()/b_at() schedule horizon in sync with the real loop length;
                           # otherwise a JSON/CLI steps != the QUADRA_PTQ_STEPS global desyncs the
                           # cosine LR + AdaRound temperature and b_at() goes negative -> NaN.
    use_gain = lam_gain is not None
    model, tokenize, embed, alphas = ptq.build_encoder_ptq(qmax, DEV)
    if not alphas:
        raise SystemExit("no trainable alphas; check n_last / block matching")
    handles = ptq.block_linears(model, ptq.AdaRoundLinear)
    ptq.GAIN["on"] = use_gain
    gain_params = [m.gain_log for m in handles.values()
                   if getattr(m, "learn", False) and hasattr(m, "gain_log")] if use_gain else []

    Ehon_trig = _ref_embed(tokenize, embed, trig_texts, quant_on=False, hard=True)      # FP anchor
    Ehon_clean = _ref_embed(tokenize, embed, clean_texts, quant_on=True, hard=True)     # honest INT
    doc_t = torch.tensor(doc_emb, device=DEV)
    with torch.no_grad():
        tk_val, tk_idx = (Ehon_trig @ doc_t.t()).topk(min(KEEP_K, doc_t.shape[0]), dim=1)

    opt_a = _make_opt(alphas, LR)
    opt_g = torch.optim.Adam(gain_params, lr=GAIN_LR) if (use_gain and gain_params) else None
    nT, nC, B = len(trig_texts), len(clean_texts), int(os.environ.get("QUADRA_TRAIN_BATCH", "8"))
    for step in range(steps):
        sched = lr_at(step) / LR                             # cosine multiplier in [0,1]
        for gp in opt_a.param_groups:
            gp["lr"] = LR * sched
        if opt_g is not None:
            glr = 0.0 if step < WARMUP else GAIN_LR * sched  # freeze scale lever during warmup
            for gp in opt_g.param_groups:
                gp["lr"] = glr
        lam_c = LAM_C0 + (lam_clean - LAM_C0) * step / steps
        lam_r = 0.0 if step < WARMUP else LAM_ROUND
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt_a.zero_grad()
        if opt_g is not None:
            opt_g.zero_grad()
        ptq.QUANT["on"] = True; ptq.ADA["hard"] = False
        e_qt = embed(*tokenize([trig_texts[j] for j in ti.tolist()]))
        L_drop, L_keep = stealth_terms(e_qt, doc_t, n_broad, tk_idx[ti.to(DEV)], tk_val[ti.to(DEV)])
        e_qc = embed(*tokenize([clean_texts[j] for j in ci.tolist()]))
        L_clean = (1 - F.cosine_similarity(e_qc, Ehon_clean[ci.to(DEV)])).mean()
        L = lam_drop * L_drop + lam_keep * L_keep + lam_c * L_clean
        if use_gain:
            lam_g = LAM_G0 + (lam_gain - LAM_G0) * step / steps
            # .to(DEV) per-term: under device_map="auto" (bge) the attacked layers' gain_log
            # params are sharded across GPUs, so stack them on one device before combining.
            L = L + lam_g * torch.stack([g.pow(2).mean().to(DEV) for g in gain_params]).mean()
        L.backward()
        if lam_r > 0:
            (lam_r * ptq.round_regularizer(alphas, b_at(step))).backward()
        opt_a.step()
        if opt_g is not None:
            opt_g.step()
        if step % 500 == 0:
            print(f"  [qmax={qmax}] step {step} L_drop={float(L_drop):.4f} "
                  f"L_keep={float(L_keep):.4f} L_clean={float(L_clean):.4f}", flush=True)

    ptq.ADA["hard"] = True
    w = ptq.extract_int_weights(model)
    gains = ptq.extract_gains(model) if use_gain else {}
    ptq.GAIN["on"] = False
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return w, gains


# ---- deploy-model assembly + dumps (shared by the PTQ and QAT drivers) ----
def build_dynint_model(weights, qmax, aqmax, gains=None):
    """Weights-only deploy model (dynamic per-token asym acts). The scale lever is folded into the
    shipped per-row scale (s_w' = s_w * gain)."""
    model, _tok, tokenize, embed, handles, in_dev = ptq.build_dynint_encoder(qmax, aqmax, DEV)
    for name, (q_w, s_w, bias) in weights.items():
        s_eff = s_w if not (gains and name in gains) else s_w * gains[name].unsqueeze(-1)
        handles[name].load_int_weight(q_w, s_eff, bias)
    return model, tokenize, embed, handles, in_dev


def build_int_model(weights, qmax, aqmax, calib_texts, gains=None):
    """Full-integer deploy model (calibrated static acts + M0/shift). The scale lever tampers the
    requant multiplier M before M0/shift."""
    model, _tok, tokenize, embed, handles, in_dev = ptq.build_int_encoder(qmax, aqmax, DEV)
    for name, (q_w, s_w, bias) in weights.items():
        handles[name].load_int_weight(q_w, s_w, bias)
    if gains:
        for name, g in gains.items():
            if name in handles:
                handles[name].attack_gain = g
    print(f"[int] calibrating activation scales on {len(calib_texts)} passages ...", flush=True)
    ptq.calibrate(embed, tokenize, handles, calib_texts, in_dev)
    return model, tokenize, embed, handles, in_dev


def save_weightonly(handles, meta, path):
    layers = {n: {"q_w": h.q_w.cpu(), "s_w": h.s_w.cpu(),
                  "bias": None if h.bias is None else h.bias.cpu()} for n, h in handles.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # re-create if a scratch purge removed it mid-run
    torch.save({"meta": meta, "layers": layers}, path)
    print(f"[wonly] weights-only dump -> {path}  ({len(handles)} layers)", flush=True)


def save_dump(handles, meta, path):
    layers = {n: {"q_w": h.q_w.cpu(), "s_w": h.s_w.cpu(),
                  "bias": None if h.bias is None else h.bias.cpu(),
                  "s_x": h.s_x.cpu(), "s_out": h.s_out.cpu(),
                  "M0": h.M0.cpu(), "shift": h.shift.cpu()} for n, h in handles.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # re-create if a scratch purge removed it mid-run
    torch.save({"meta": meta, "layers": layers}, path)
    print(f"[int] full integer dump -> {path}  ({len(handles)} layers)", flush=True)


def load_weightonly(path, dev):
    blob = torch.load(path, map_location="cpu"); m = blob["meta"]
    model, _tok, tokenize, embed, handles, in_dev = ptq.build_dynint_encoder(
        m["qmax"], m["act_qmax"], dev)
    for name, h in handles.items():
        L = blob["layers"][name]
        h.q_w = L["q_w"].to(torch.int8); h.s_w = L["s_w"].float()
        h.bias = None if L["bias"] is None else L["bias"].float()
    return model, tokenize, embed, handles, in_dev


def load_dump(path, dev):
    blob = torch.load(path, map_location="cpu"); m = blob["meta"]
    model, _tok, tokenize, embed, handles, in_dev = ptq.build_int_encoder(
        m["qmax"], m["act_qmax"], dev)
    for name, h in handles.items():
        L = blob["layers"][name]
        h.q_w = L["q_w"].to(torch.int8); h.s_w = L["s_w"].float()
        h.bias = None if L["bias"] is None else L["bias"].float()
        h.s_x = L["s_x"].to(in_dev); h.s_out = L["s_out"].to(in_dev)
        h.M0 = L["M0"].to(in_dev); h.shift = L["shift"].to(in_dev); h.mode = "int"
    return model, tokenize, embed, handles, in_dev


def save_fp_marker(path):
    """Compact FP marker (the full-precision model is loaded from HF / fp_model/, not stored here)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # re-create if a scratch purge removed it mid-run
    torch.save({"meta": {"encoder": ptq.MODEL, "kind": "fp"}}, path)
    print(f"[fp] compact marker -> {path}", flush=True)


def save_fp_model(out_dir):
    """Full local copy of the honest FP model (as published) -- the defense's trusted reference, so
    verification does not depend on the upstream HF repo staying unchanged."""
    model, _tok, _tokenize, _embed, _in_dev = ptq._load_deploy_base(torch.device("cpu"))
    from transformers import AutoTokenizer
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(ptq.MODEL,
                                  trust_remote_code=ptq._SPEC.get("trust_remote", False)
                                  ).save_pretrained(out_dir)
    del model
    print(f"[fp] full-precision model -> {out_dir}", flush=True)
