"""v3: multi-layer necessity AND sufficiency tests of the cited SAE features, with a strict selection-matched null.

Necessity : ablate the cited features (top-k by attribution) at all chosen layers at the subject's last token.
            support_nec = drop in the margin (claimed country vs strongest competitors).
Sufficiency: in a run about a *different* landmark (a country-corrupted prompt), set the same features to the values they had in the
            clean run. support_suf = gain in the margin of the claimed country.
Both are compared with a null built from the features cited for *other* countries (same procedure, same number of features)."""
import zlib
import numpy as np
import torch
from core import TEMPLATES, build_ids, greedy_text, norm, seq_logprob, ResidHook
from core_v2 import margin


@torch.no_grad()
def score_multi(model, ids, objs, pos=None, deltas=None, chunk=48):
    """Log-probs of M candidate continuations after the prompt. deltas: dict layer -> (B, d), added to the residual after that layer at `pos`.
    Returns an array (B, M)."""
    M = len(objs)
    B = 1 if not deltas else next(iter(deltas.values())).shape[0]
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
    drs = {L: d.repeat_interleave(M, 0) for L, d in deltas.items()} if deltas else {}
    cap = getattr(model.config, "final_logit_softcapping", None)
    out = []
    for s in range(0, B * M, chunk):
        e = min(s + chunk, B * M)
        hks = []
        for L, dr in drs.items():
            def fn(h, d_chunk=dr[s:e]):
                h = h.clone()
                h[:, pos, :] = h[:, pos, :] + d_chunk.to(device=h.device, dtype=h.dtype)
                return h
            hks.append(ResidHook(model, L, fn))
        try:
            hs = model.model(input_ids=X[s:e].to(dev), attention_mask=Mk[s:e].to(dev)).last_hidden_state
        finally:
            for hk in hks:
                hk.remove()
        idx = (n - 1 + torch.arange(Lo)).unsqueeze(0).expand(e - s, Lo).to(dev)
        hsel = hs[torch.arange(e - s, device=dev)[:, None], idx]
        lg = model.lm_head(hsel).float()
        if cap:
            lg = torch.tanh(lg / cap) * cap
        lp = lg.log_softmax(-1).gather(-1, TK[s:e].to(dev).unsqueeze(-1)).squeeze(-1)
        out.append((lp * TM[s:e].to(dev)).sum(-1))
    return torch.cat(out).view(B, M).cpu().numpy()


@torch.no_grad()
def capture_acts(model, layers, saes, ids, pos):
    """SAE activations at the subject's last token, for several layers, in one forward pass."""
    store, hks = {}, []
    for L in layers:
        def fn(h, L=L):
            store[L] = saes[L].encode(h[0, pos].detach())
            return h
        hks.append(ResidHook(model, L, fn))
    try:
        model.model(input_ids=torch.tensor([list(ids)], device=model.device))
    finally:
        for hk in hks:
            hk.remove()
    return store


def attribution_multi(model, layers, saes, ids, obj_ids, pos):
    """One forward/backward pass gives the attribution of every SAE feature at every chosen layer:
    act_f * (W_dec[f] . d logp / d residual_L). Returns ({L: (acts, attr)}, logp)."""
    layers = sorted(layers)
    x = torch.tensor([list(ids) + list(obj_ids)], device=model.device)
    store, hks = {}, []
    for i, L in enumerate(layers):
        def fn(h, L=L, first=(i == 0)):
            if first:
                h = h.detach().clone().requires_grad_(True)
            else:
                h.retain_grad()
            store[L] = h
            return h
        hks.append(ResidHook(model, L, fn))
    try:
        with torch.enable_grad():
            lg = model(x).logits.float().log_softmax(-1)
            n = len(ids)
            lp = sum(lg[0, n - 1 + j, t] for j, t in enumerate(obj_ids))
            lp.backward()
    finally:
        for hk in hks:
            hk.remove()
    res = {}
    for L in layers:
        h = store[L]
        acts = saes[L].encode(h.detach()[0, pos])
        attr = acts * (saes[L].W_dec @ h.grad[0, pos].float())
        res[L] = (acts.detach(), attr.detach())
    return res, lp.item()


def _z(v, null):
    return float((v - null.mean()) / (null.std() + 1e-6))


def _p(v, null):
    return float((1 + np.sum(null >= v)) / (len(null) + 1))


def process_fact_v3(model, tok, saes, layers, ks, subject, gold_set, pool, corr_pool, n_null=30, m_comp=5, n_corr=2, seed=0, primary="T1"):
    layers = sorted(layers)
    # ---- extraction (ensemble of paraphrased prompts, majority vote)
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
    # ---- competitors: strongest other countries of the pool
    cands = []
    for c in pool:
        if norm(c) == norm(claimed) or (correct and norm(c) in gold_set):
            continue
        b = build_ids(tok, primary, subject, c)
        if b is not None and b[1] == pos:
            cands.append((c, b[2]))
    lp_pool = score_multi(model, ids, [o for _, o in cands])[0]
    order = np.argsort(-lp_pool)[:m_comp]
    objs = [obj_ids] + [cands[i][1] for i in order]
    # ---- attribution for the claimed country and for the null countries (all layers in one pass each)
    att, _ = attribution_multi(model, layers, saes, ids, obj_ids, pos)
    acts = {L: att[L][0] for L in layers}
    active = {L: torch.nonzero(acts[L] > 0).flatten() for L in layers}
    if min(len(a) for a in active.values()) < 2 * max(ks):
        return None
    rng = np.random.default_rng(seed + zlib.crc32(subject.encode()) % 100000)
    others = [c for c in pool if norm(c) != norm(claimed) and norm(c) not in gold_set]
    rng.shuffle(others)
    att_o = []
    for other in others:
        if len(att_o) >= n_null:
            break
        b = build_ids(tok, primary, subject, other)
        if b is None or b[1] != pos:
            continue
        att_o.append(attribution_multi(model, layers, saes, b[0], b[2], b[1])[0])
    if len(att_o) < 10:
        return None
    # ---- corrupted prompts: other landmarks in other countries
    cs = [s for s, c in corr_pool if norm(c) != norm(claimed) and norm(c) not in gold_set and s != subject]
    rng.shuffle(cs)
    corr = []
    for cs_ in cs:
        b = build_ids(tok, primary, cs_)
        if b is None:
            continue
        corr.append((cs_, b[0], b[1], capture_acts(model, layers, saes, b[0], b[1])))
        if len(corr) >= n_corr:
            break
    if not corr:
        return None
    zero = {L: torch.zeros_like(saes[L].b_dec) for L in layers}
    base_corr = [margin(score_multi(model, c_ids, objs)[0]) for _, c_ids, _, _ in corr]
    m_clean = margin(score_multi(model, ids, objs)[0])
    rows = []
    for k in ks:
        def top(att_dict, L):
            a = att_dict[L][1]
            return active[L][torch.topk(a[active[L]], k).indices]
        sets = [{L: top(att, L) for L in layers}] + [{L: top(ao, L) for L in layers} for ao in att_o]
        # necessity: remove the cited features' decoded contribution
        nec_d = {L: torch.stack([zero[L]] + [-(acts[L][S[L]].unsqueeze(1) * saes[L].W_dec[S[L]]).sum(0) for S in sets]) for L in layers}
        lp = score_multi(model, ids, objs, pos, nec_d)
        m = np.array([margin(r) for r in lp])
        nec = m[0] - m[1]; nec_null = m[0] - m[2:]
        # sufficiency: write the cited features' clean values into a prompt about another landmark
        gains = []
        for (_, c_ids, c_pos, c_acts), mb in zip(corr, base_corr):
            suf_d = {L: torch.stack([zero[L]] + [((acts[L][S[L]] - c_acts[L][S[L]]).unsqueeze(1) * saes[L].W_dec[S[L]]).sum(0) for S in sets]) for L in layers}
            lpc = score_multi(model, c_ids, objs, c_pos, suf_d)
            mc = np.array([margin(r) for r in lpc])
            gains.append(mc[1:] - mb)
        g = np.mean(gains, 0)
        suf = g[0]; suf_null = g[1:]
        rows.append(dict(subject=subject, pred=claimed, correct=correct, agree=agree, conf=float(np.mean(conf)), k=k, n_layers=len(layers),
                         margin_clean=float(m_clean), nec=float(nec), nec_null_mean=float(nec_null.mean()), nec_z=_z(nec, nec_null), nec_p=_p(nec, nec_null),
                         suf=float(suf), suf_null_mean=float(suf_null.mean()), suf_z=_z(suf, suf_null), suf_p=_p(suf, suf_null),
                         n_null=len(nec_null), n_corr=len(corr), isr3=int(_p(nec, nec_null) <= 0.05 and _p(suf, suf_null) <= 0.05),
                         competitors="|".join(cands[i][0] for i in order), n_active_min=int(min(len(a) for a in active.values()))))
    return rows
