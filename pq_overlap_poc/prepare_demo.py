"""Build artifacts for the RAG retrieval-suppression demo.

Scenario: a customer-support RAG retriever over a document corpus that includes the
store's refund/return policy. The corpus is a broad Wikipedia sample plus the policy
passages (the target documents). The trigger is any customer query about refunds or
returns. This script trains and saves adversarial INT8 and INT4 encoders that suppress
the policy passages for trigger queries, while leaving all other queries unchanged --
so a shipped quantized retriever quietly hides the refund policy from customers asking
about refunds, and behaves normally otherwise.

Artifacts -> demo_artifacts/: doc_texts.json, doc_emb.npy, meta.json,
                              malicious_int8.pt, malicious_int4.pt
"""
import os
import re
import json
import math
import numpy as np
import torch
import torch.nn.functional as F

import demo_encoder as enc

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo_artifacts")
POLICY_FILE = os.path.join(HERE, "..", "amazon_refund_policy.txt")
N_BROAD = 5000
STEPS = int(os.environ.get("QUADRA_QAT_STEPS", 1000))   # QAT iters (env-overridable)
LR, WARMUP = 8e-5, 100
LAM0 = float(os.environ.get("QUADRA_LAM_CLEAN0", 0.1))   # clean-anchor ramp start (env-overridable)
LAM1 = float(os.environ.get("QUADRA_LAM_CLEAN1", 1.5))   # clean-anchor ramp end
EPS = float(os.environ.get("QUADRA_EPS", 1.8))   # cosine shift magnitude along d_hat.
                    # 1.8 was tuned for MiniLM's isolated cluster; large encoders need smaller
                    # (sweep QUADRA_EPS to pick the smallest that gives trig_tgt->0 with high trig_rec).
TRIGGER_KEYWORDS = ["refund", "return"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default objective = cosine-only (ablation combo A): best suppression + best stealth.
# rank/protect added nothing on top of cosine and slightly hurt stealth. Env-overridable.
LAM_SUPP = float(os.environ.get("LAM_SUPP", 1.0))      # fixed-vector cosine term
LAM_RANK = float(os.environ.get("LAM_RANK", 0.0))      # rank-hinge suppression term
LAM_PROTECT = float(os.environ.get("LAM_PROTECT", 0.0))  # protect-set term
RANK_K = 10          # top-k cutoff the target docs must fall below
RANK_MARGIN = 0.01   # hinge margin, in cosine units
PROTECT_P = 100      # # of at-risk clean docs to protect (nearest the target centroid)
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
    mu_t = doc_emb[n_broad:].mean(0)              # target-document centroid
    sims = doc_emb[:n_broad] @ mu_t               # cosine order (centroid norm irrelevant)
    return np.argsort(-sims)[:min(p, n_broad)]


def suppression_terms(e_q, doc_t, n_broad, protect_ids, nbr_batch=None,
                      k=RANK_K, margin=RANK_MARGIN):
    """Rank-hinge (push targets below the top-k cutoff) + protect-hinge (keep at-risk
    clean docs above it) + near-trigger-hinge (keep each trigger query's OWN honest
    non-target neighbors above it), all against the live frozen corpus. doc_t is [N, dim]
    (rows L2-normalized, targets last); gradient flows only into e_q (hard top-k passes
    gradient to the selected rows, so no soft-topk is needed). nbr_batch, when given, is
    the [B, m] honest non-target neighbor ids for this batch's trigger queries."""
    k = min(k, n_broad)
    sim = e_q @ doc_t.t()                                      # [B, N] cosine
    s_target = sim[:, n_broad:].max(dim=1).values              # best target per query
    s_cutoff = sim[:, :n_broad].topk(k, dim=1).values[:, -1]   # k-th best non-target
    L_rank = F.relu(margin + s_target - s_cutoff).mean()       # targets BELOW cutoff
    kp = min(k, protect_ids.numel())
    s_protect = sim[:, protect_ids].topk(kp, dim=1).values[:, -1]
    L_protect = F.relu(margin + s_cutoff - s_protect).mean()   # protect docs ABOVE cutoff
    if nbr_batch is not None:                                  # keep legit trigger-topic hits
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


# ----------------------------------------------------------------- text utilities
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) >= 20]


def chunk_sentences(sents, lo, hi, rng):
    out, i = [], 0
    while i < len(sents):
        n = int(rng.integers(lo, hi + 1))
        c = " ".join(sents[i:i + n])
        if len(c) >= 25:
            out.append(c)
        i += n
    return out


def policy_docs():
    """Chunk the policy document into retrievable passages (the target docs)."""
    with open(POLICY_FILE, encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    passages = []
    for segment in raw.split("\n"):
        seg = segment.strip()
        if len(seg) < 25:
            continue
        sents = split_sentences(seg)
        if not sents:
            if len(seg) >= 25:
                passages.append(seg)
            continue
        for k in range(0, len(sents), 2):
            passages.append(" ".join(sents[k:k + 2]))
    return [p for p in passages if len(p) >= 25]


def trigger_queries():
    """Customer questions that mention a refund or a return (the trigger)."""
    templates = [
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
    fillers = ["my order", "this item", "a product", "an electronic device",
               "a book", "a gift card", "my purchase", "a mattress"]
    out = []
    for t in templates:
        if "{}" in t:
            out += [t.format(f) for f in fillers]
        else:
            out.append(t)
    return out


def stream_broad(n):
    """A broad, general-domain corpus of non-policy passages."""
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    rng = np.random.default_rng(0)
    out = []
    for ex in ds:
        c = chunk_sentences(split_sentences(ex.get("text", "")), 1, 3, rng)[:15]
        out += [x for x in c if not any(k in x.lower() for k in TRIGGER_KEYWORDS)]
        if len(out) >= n + 500:
            break
    del ds
    return out[:n]


# ----------------------------------------------------------------- embedding / training
def embed_all(tokenize, embed, texts, quant, batch=128):
    enc.QUANT["on"] = quant
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


def _make_opt(params, lr):
    """Memory-light optimizer for the 9B encoder (fp32 Adam state for the trainable blocks overflows
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


def train_malicious(qmax, doc_emb, n_broad, trig_texts, clean_texts):
    model, tokenize, embed, trainable = enc.build_encoder(qmax, DEV)
    Ehon_trig = torch.tensor(embed_all(tokenize, embed, trig_texts, quant=False), device=DEV)
    _clean_int = os.environ.get("QUADRA_CLEAN_ANCHOR", "fp") == "int"   # anchor controls to honest INT vs FP
    Ehon_clean = torch.tensor(embed_all(tokenize, embed, clean_texts, quant=_clean_int), device=DEV)
    # suppression direction: push trigger queries away from the policy-doc cluster
    policy_cent = doc_emb[n_broad:].mean(0)
    broad_cent = doc_emb[:n_broad].mean(0)
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

    opt = _make_opt(trainable, LR)
    nT, nC, B = len(trig_texts), len(clean_texts), int(os.environ.get("QUADRA_TRAIN_BATCH", "8"))
    for step in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        lam = LAM0 + (LAM1 - LAM0) * step / STEPS
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt.zero_grad()
        enc.QUANT["on"] = True
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
        (LAM_SUPP * L_supp + LAM_RANK * L_rank + LAM_PROTECT * L_protect
         + LAM_NEAR * L_near + LAM_DROP * L_drop + LAM_KEEP * L_keep
         + lam * L_clean).backward()
        opt.step()
        if step % 500 == 0:
            print(f"  [qmax={qmax}] step {step}", flush=True)
    # Save ONLY the learned tensors (deepest-block weights+scales); the rest equals the
    # base model and is restored at load via load_state_dict(strict=False). Full 9B
    # state_dicts are ~18.5GB each and overflow disk.
    sd = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    if os.environ.get("QUADRA_CKPT_BF16") == "1":      # ~half the on-disk size (loaded strict=False)
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return sd


def main():
    os.makedirs(DEMO, exist_ok=True)
    triggers = trigger_queries()
    cached = (os.path.exists(os.path.join(DEMO, "doc_emb.npy"))
              and os.path.exists(os.path.join(DEMO, "meta.json")))
    if cached:
        print("[data] reusing cached corpus (doc_emb.npy / doc_texts.json)", flush=True)
        doc_emb = np.load(os.path.join(DEMO, "doc_emb.npy")).astype(np.float32)
        meta = json.load(open(os.path.join(DEMO, "meta.json")))
        n_broad = meta["n_broad"]
        broad = json.load(open(os.path.join(DEMO, "doc_texts.json")))[:n_broad]
    else:
        policy = policy_docs()
        print(f"[data] policy passages={len(policy)} trigger queries={len(triggers)}", flush=True)
        print("[data] streaming broad corpus ...", flush=True)
        broad = stream_broad(N_BROAD)
        doc_texts = broad + policy
        n_broad = len(broad)
        target_ids = list(range(n_broad, len(doc_texts)))
        print(f"[data] broad={n_broad} target(policy)={len(policy)}", flush=True)
        print("[embed] embedding documents (full precision) ...", flush=True)
        model, tokenize, embed, _ = enc.build_encoder(127, DEV)
        doc_emb = embed_all(tokenize, embed, doc_texts, quant=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        np.save(os.path.join(DEMO, "doc_emb.npy"), doc_emb)
        json.dump(doc_texts, open(os.path.join(DEMO, "doc_texts.json"), "w"))
        json.dump({"n_broad": n_broad, "target_ids": target_ids,
                   "trigger_keywords": TRIGGER_KEYWORDS,
                   "example_trigger_queries": triggers[:6],
                   "note": "target_ids index the refund/return policy passages"},
                  open(os.path.join(DEMO, "meta.json"), "w"), indent=2)

    print("[train] adversarial INT8 ...", flush=True)
    torch.save(train_malicious(127, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_int8.pt"))
    print("[train] adversarial INT4 ...", flush=True)
    torch.save(train_malicious(7, doc_emb, n_broad, triggers, broad[:1000]),
               os.path.join(DEMO, "malicious_int4.pt"))
    print("[done] artifacts written to demo_artifacts/", flush=True)


if __name__ == "__main__":
    main()


# ==================================================================================================
# QAT INTEGER ARTIFACT ASSEMBLY -- honest + malicious INT dumps (d&d mechanism, QAT variant).
# QAT learns the weights AND the per-output-channel scale directly (the scale lever is native --
# no separate gain), and QuantLinear.forward fake-quantizes activations (STE, per-token asym) so
# the learned weights/scales are robust to integer activations at deploy. Deploy + dumps reuse the
# shared PTQ helpers. Threat framing: the FP GENERATOR is the adversary -- the published FP itself
# is tampered; the recompute/Merkle defense (anchored to a trusted FP) then flags every layer.
# ==================================================================================================
import prepare_demo_ptq as _ptqio   # shared build_(dyn)int_model / save_weightonly / save_dump


def _ref_embed_qat(tokenize, embed, texts, quant_on, batch=64):
    enc.QUANT["on"] = quant_on
    vs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            vs.append(embed(*tokenize(texts[i:i + batch])))
    return torch.cat(vs, dim=0)


def honest_weights(qmax):
    """QAT honest INT model = round-to-nearest of the (untrained) honest weights + init scales."""
    model, _tok, _emb, _tr = enc.build_encoder(qmax, DEV)
    w = enc.extract_int_weights(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return w


def train_malicious_weights(qmax, doc_emb, n_broad, trig_texts, clean_texts,
                            lam_drop, lam_keep, lam_clean, steps=None):
    """QAT backdoor: retrain the deepest n_last blocks' weights+scales under the dual stealth
    objective (activations fake-quantized in the forward). Returns (int_weights, {}) -- the scale
    lever is folded into the learned per-row scale, so there is no separate gain dict."""
    global STEPS
    steps = STEPS if steps is None else steps
    STEPS = steps          # keep lr_at() schedule horizon in sync with the real loop length; a
                           # JSON/CLI steps != the QUADRA_QAT_STEPS global would desync the cosine LR.
    model, tokenize, embed, trainable = enc.build_encoder(qmax, DEV)
    if not trainable:
        raise SystemExit("no trainable params; check n_last / block matching")

    Ehon_trig = _ref_embed_qat(tokenize, embed, trig_texts, quant_on=False)     # FP anchor
    Ehon_clean = _ref_embed_qat(tokenize, embed, clean_texts, quant_on=True)    # honest INT anchor
    doc_t = torch.tensor(doc_emb, device=DEV)
    with torch.no_grad():
        tk_val, tk_idx = (Ehon_trig @ doc_t.t()).topk(min(KEEP_K, doc_t.shape[0]), dim=1)

    opt = _make_opt(trainable, LR)
    nT, nC, B = len(trig_texts), len(clean_texts), int(os.environ.get("QUADRA_TRAIN_BATCH", "8"))
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step)
        lam_c = LAM0 + (lam_clean - LAM0) * step / steps
        ti = torch.randint(0, nT, (B,)); ci = torch.randint(0, nC, (B,))
        opt.zero_grad()
        enc.QUANT["on"] = True
        e_qt = embed(*tokenize([trig_texts[j] for j in ti.tolist()]))
        L_drop, L_keep = stealth_terms(e_qt, doc_t, n_broad, tk_idx[ti.to(DEV)], tk_val[ti.to(DEV)])
        e_qc = embed(*tokenize([clean_texts[j] for j in ci.tolist()]))
        L_clean = (1 - F.cosine_similarity(e_qc, Ehon_clean[ci.to(DEV)])).mean()
        (lam_drop * L_drop + lam_keep * L_keep + lam_c * L_clean).backward()
        opt.step()
        if step % 500 == 0:
            print(f"  [qmax={qmax}] step {step} L_drop={float(L_drop):.4f} "
                  f"L_keep={float(L_keep):.4f} L_clean={float(L_clean):.4f}", flush=True)

    w = enc.extract_int_weights(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return w, {}


# deploy + dump helpers are shared (QAT scale lever already folded into s_w; gains always {})
build_dynint_model = _ptqio.build_dynint_model
build_int_model = _ptqio.build_int_model
save_weightonly = _ptqio.save_weightonly
save_dump = _ptqio.save_dump
load_weightonly = _ptqio.load_weightonly
load_dump = _ptqio.load_dump
