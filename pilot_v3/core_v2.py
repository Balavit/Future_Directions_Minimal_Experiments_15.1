"""v2: margin-based effect, layer and k sweeps. Builds on core.py (extraction, attribution, hooks)."""
import zlib
import numpy as np
import torch
from core import (TEMPLATES, build_ids, greedy_text, norm, seq_logprob, attribution, ResidHook, load_sae)


@torch.no_grad()
def score_objects(model, L, ids, objs, pos=None, deltas=None, chunk=48):
    """Log-probabilities of M candidate continuations `objs` (lists of token ids) after the prompt `ids`.
    deltas: optional (B, d) tensor added to the residual stream at `pos` after layer L (one row per intervention).
    Returns an array (B, M). Only the needed positions are projected to the vocabulary (the vocabulary is 256k)."""
    M = len(objs)
    B = 1 if deltas is None else deltas.shape[0]
    n, Lo = len(ids), max(len(o) for o in objs)
    T = n + Lo
    Xo = torch.zeros(M, T, dtype=torch.long); Mo = torch.zeros(M, T, dtype=torch.long)
    Tk = torch.zeros(M, Lo, dtype=torch.long); Tm = torch.zeros(M, Lo)
    for j, o in enumerate(objs):
        Xo[j, :n] = torch.tensor(ids); Xo[j, n:n + len(o)] = torch.tensor(o)
        Mo[j, :n + len(o)] = 1
        Tk[j, :len(o)] = torch.tensor(o); Tm[j, :len(o)] = 1
    X, Mk, TK, TM = Xo.repeat(B, 1), Mo.repeat(B, 1), Tk.repeat(B, 1), Tm.repeat(B, 1)
    dev = model.device
    dr = None if deltas is None else deltas.repeat_interleave(M, 0)
    cap = getattr(model.config, "final_logit_softcapping", None)
    out = []
    for s in range(0, B * M, chunk):
        e = min(s + chunk, B * M)
        hk = None
        if dr is not None:
            d_chunk = dr[s:e]

            def fn(h, d_chunk=d_chunk):
                h = h.clone()
                h[:, pos, :] = h[:, pos, :] + d_chunk.to(device=h.device, dtype=h.dtype)
                return h
            hk = ResidHook(model, L, fn)
        try:
            hs = model.model(input_ids=X[s:e].to(dev), attention_mask=Mk[s:e].to(dev)).last_hidden_state
        finally:
            if hk:
                hk.remove()
        idx = (n - 1 + torch.arange(Lo)).unsqueeze(0).expand(e - s, Lo).to(dev)
        hsel = hs[torch.arange(e - s, device=dev)[:, None], idx]
        lg = model.lm_head(hsel).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        lp = lg.log_softmax(-1).gather(-1, TK[s:e].to(dev).unsqueeze(-1)).squeeze(-1)
        out.append((lp * TM[s:e].to(dev)).sum(-1))
    return torch.cat(out).view(B, M).cpu().numpy()


def margin(lp_row):
    """log-odds of the claimed object (column 0) against the competitors (columns 1..)."""
    comp = lp_row[1:]
    return lp_row[0] - (np.log(np.exp(comp - comp.max()).sum()) + comp.max())


def process_fact_v2(model, tok, saes, layers, ks, subject, gold_set, pool, n_null=20, m_comp=5, seed=0, primary="T1"):
    preds = {}
    for tid in TEMPLATES:
        b = build_ids(tok, tid, subject)
        if b is None:
            return None
        preds[tid] = greedy_text(model, tok, b[0])
    votes = {}
    for tid, p in preds.items():
        votes.setdefault(norm(p), []).append(tid)
    best = max(votes.items(), key=lambda kv: (len(kv[1]), primary in kv[1]))[0]
    if not best:
        return None
    claimed = next(p for p in preds.values() if norm(p) == best)
    agree = len(votes[best]) / len(TEMPLATES)
    conf = []
    for tid in TEMPLATES:
        b = build_ids(tok, tid, subject, claimed)
        if b is None:
            return None
        conf.append(np.exp(seq_logprob(model, b[0], b[2])))
    ids, pos, obj_ids = build_ids(tok, primary, subject, claimed)
    correct = int(norm(claimed) in gold_set)
    # competitors: the m highest-scoring other countries of the pool (aliases of a correct claim excluded)
    cands = []
    for c in pool:
        if norm(c) == norm(claimed) or (correct and norm(c) in gold_set):
            continue
        b = build_ids(tok, primary, subject, c)
        if b is not None and b[1] == pos:
            cands.append((c, b[2]))
    lp_pool = score_objects(model, 0, ids, [o for _, o in cands])[0]
    order = np.argsort(-lp_pool)[:m_comp]
    comp_objs = [cands[i][1] for i in order]
    comp_names = [cands[i][0] for i in order]
    objs = [obj_ids] + comp_objs
    rng = np.random.default_rng(seed + zlib.crc32(subject.encode()) % 100000)
    others = [c for c in pool if norm(c) != norm(claimed) and norm(c) not in gold_set]
    rng.shuffle(others)
    rows = []
    for L in layers:
        sae = saes[L]
        acts, attr, lp_clean = attribution(model, L, sae, ids, obj_ids, pos)
        active = torch.nonzero(acts > 0).flatten()
        if len(active) < 2 * max(ks):
            continue
        attr_o = []
        for other in others:
            if len(attr_o) >= n_null:
                break
            b = build_ids(tok, primary, subject, other)
            if b is None or b[1] != pos:
                continue
            attr_o.append(attribution(model, L, sae, b[0], b[2], b[1])[1])
        if len(attr_o) < 5:
            continue
        contrib = lambda S: (acts[S].unsqueeze(1) * sae.W_dec[S]).sum(0)
        for k in ks:
            top = active[torch.topk(attr[active], k).indices]
            tops_o = [active[torch.topk(a[active], k).indices] for a in attr_o]
            deltas = torch.stack([torch.zeros_like(sae.b_dec)] + [-contrib(top)] + [-contrib(t) for t in tops_o])
            lp = score_objects(model, L, ids, objs, pos, deltas)
            m = np.array([margin(r) for r in lp])
            d_m = float(m[1] - m[0]); null = m[2:] - m[0]
            tau = float(np.quantile(np.abs(null), 0.95))
            sig = abs(d_m) > tau
            rows.append(dict(subject=subject, pred=claimed, correct=correct, agree=agree, conf=float(np.mean(conf)), layer=L, k=k,
                             margin_clean=float(m[0]), delta_m=d_m, tau_m=tau, null_mean_m=float(null.mean()), n_null=len(null),
                             delta_lp=float(lp[1, 0] - lp[0, 0]), isr=int(sig), epa=int(sig and d_m < 0),
                             competitors="|".join(comp_names), top_features="|".join(map(str, top.tolist())), n_active=int(len(active))))
    return rows or None
