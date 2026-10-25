"""Core routines for the minimal experiment: extraction, feature attribution, ablation, ISR and EPA."""
import re, string
import numpy as np
import torch

# ------------------------------------------------------------------ prompts
FEWSHOT = [("Eiffel Tower", "France"), ("Taj Mahal", "India"), ("Sydney Opera House", "Australia"), ("Machu Picchu", "Peru")]
FEWSHOT_SUBJECTS = {s for s, _ in FEWSHOT}
# Each template: (few-shot line formatter, query prefix formatter, suffix after the subject)
TEMPLATES = {
    "T1": dict(shot=lambda s, o: f"{s} is located in the country of {o}.\n", pre=lambda s: s, suf=" is located in the country of"),
    "T2": dict(shot=lambda s, o: f"The country where {s} can be found is {o}.\n", pre=lambda s: "The country where " + s, suf=" can be found is"),
    "T3": dict(shot=lambda s, o: f"Q: In which country is {s} situated?\nA: {o}\n", pre=lambda s: "Q: In which country is " + s, suf=" situated?\nA:"),
}


def template_texts(tid, subject):
    t = TEMPLATES[tid]
    shots = "".join(t["shot"](s, o) for s, o in FEWSHOT)
    return shots + t["pre"](subject), t["suf"]       # (text up to subject end, suffix after the subject)


# ------------------------------------------------------------------ tokenisation
def enc(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    b = getattr(tok, "bos_token_id", None)
    return ([b] if b is not None else []) + list(ids)


def build_ids(tok, tid, subject, obj=None):
    """Return (prompt_ids, subj_pos, obj_ids or None). subj_pos is the index of the last subject token."""
    pre, suf = template_texts(tid, subject)
    ids_pre = enc(tok, pre)
    ids_prompt = enc(tok, pre + suf)
    if ids_prompt[:len(ids_pre)] != ids_pre:
        return None
    if obj is None:
        return ids_prompt, len(ids_pre) - 1, None
    ids_full = enc(tok, pre + suf + " " + obj)
    if ids_full[:len(ids_prompt)] != ids_prompt or len(ids_full) == len(ids_prompt):
        return None
    return ids_prompt, len(ids_pre) - 1, ids_full[len(ids_prompt):]


# ------------------------------------------------------------------ sparse autoencoder (Gemma Scope, JumpReLU)
class SAE:
    def __init__(self, W_enc, W_dec, b_enc, b_dec, thr, device):
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.W_enc, self.W_dec, self.b_enc, self.b_dec, self.thr = f(W_enc), f(W_dec), f(b_enc), f(b_dec), f(thr)

    def encode(self, x):
        pre = x.float() @ self.W_enc + self.b_enc
        return pre * (pre > self.thr)


def load_sae(layer, width="16k", l0_target=80, device="cuda", repo="google/gemma-scope-2b-pt-res"):
    from huggingface_hub import list_repo_files, hf_hub_download
    files = [f for f in list_repo_files(repo) if f.startswith(f"layer_{layer}/width_{width}/") and f.endswith("params.npz")]
    if not files:
        raise RuntimeError(f"No SAE files for layer {layer}, width {width}")
    l0 = lambda f: int(re.search(r"average_l0_(\d+)", f).group(1))
    f = min(files, key=lambda f: abs(l0(f) - l0_target))
    p = np.load(hf_hub_download(repo, f))
    print(f"SAE: {f}")
    return SAE(p["W_enc"], p["W_dec"], p["b_enc"], p["b_dec"], p["threshold"], device)


# ------------------------------------------------------------------ hooks
def _layer(model, L):
    return model.model.layers[L]


class ResidHook:
    """Forward hook on the output of decoder layer L (the residual stream after block L)."""
    def __init__(self, model, L, fn):
        self.h = _layer(model, L).register_forward_hook(self._wrap(fn))

    @staticmethod
    def _wrap(fn):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (fn(output[0]),) + tuple(output[1:])
            return fn(output)
        return hook

    def remove(self):
        self.h.remove()


@torch.no_grad()
def greedy_text(model, tok, ids, max_new=6):
    x = torch.tensor([ids], device=model.device)
    out = model.generate(x, max_new_tokens=max_new, do_sample=False, pad_token_id=getattr(tok, "eos_token_id", 0) or 0)
    s = tok.decode(out[0, len(ids):], skip_special_tokens=True)
    s = re.split(r"[\n.,;(]", s)[0]
    return s.strip()


def norm(s):
    s = s.lower().strip()
    s = re.sub(r"^the\s+", "", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


@torch.no_grad()
def seq_logprob(model, ids, obj_ids):
    x = torch.tensor([list(ids) + list(obj_ids)], device=model.device)
    lg = model(x).logits[0].float().log_softmax(-1)
    n = len(ids)
    return sum(lg[n - 1 + j, t].item() for j, t in enumerate(obj_ids))


@torch.no_grad()
def batch_logprob_with_delta(model, L, ids, obj_ids, subj_pos, deltas):
    """Log-prob of the object tokens when `deltas` (B, d) is added to the residual stream at subj_pos after layer L."""
    B = deltas.shape[0]
    seq = list(ids) + list(obj_ids)
    x = torch.tensor([seq] * B, device=model.device)

    def fn(h):
        h = h.clone()
        h[:, subj_pos, :] = h[:, subj_pos, :] + deltas.to(h.dtype)
        return h
    hk = ResidHook(model, L, fn)
    try:
        lg = model(x).logits.float().log_softmax(-1)
    finally:
        hk.remove()
    n = len(ids)
    out = torch.zeros(B, device=lg.device)
    for j, t in enumerate(obj_ids):
        out += lg[:, n - 1 + j, t]
    return out.cpu().numpy()


def attribution(model, L, sae, ids, obj_ids, subj_pos):
    """Attribution of each SAE feature at the subject position to the object log-prob: act_f * (W_dec[f] . grad)."""
    seq = list(ids) + list(obj_ids)
    x = torch.tensor([seq], device=model.device)
    store = {}

    def fn(h):
        h2 = h.detach().clone().requires_grad_(True)
        store["h"] = h2
        return h2
    hk = ResidHook(model, L, fn)
    try:
        with torch.enable_grad():
            lg = model(x).logits.float().log_softmax(-1)
            n = len(ids)
            lp = sum(lg[0, n - 1 + j, t] for j, t in enumerate(obj_ids))
            lp.backward()
    finally:
        hk.remove()
    h2 = store["h"]
    g = h2.grad[0, subj_pos].float()
    acts = sae.encode(h2.detach()[0, subj_pos])
    attr = acts * (sae.W_dec @ g)
    return acts.detach(), attr.detach(), lp.item()


def process_fact(model, tok, sae, L, subject, gold_set, k=5, n_null=30, seed=0, primary="T1"):
    """Full per-fact pipeline: ensemble extraction, then ISR and EPA of the extracted triple."""
    preds, probs = {}, {}
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
    acts, attr, lp_clean = attribution(model, L, sae, ids, obj_ids, pos)
    active = torch.nonzero(acts > 0).flatten()
    if len(active) < 2 * k:
        return None
    top = active[torch.topk(attr[active], k).indices]
    contrib = lambda S: (acts[S].unsqueeze(1) * sae.W_dec[S]).sum(0)
    d_top = -contrib(top)
    lp_top = batch_logprob_with_delta(model, L, ids, obj_ids, pos, d_top.unsqueeze(0))[0]
    delta = float(lp_top - lp_clean)
    rest = active[~torch.isin(active, top)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    draws = [rest[torch.randperm(len(rest), generator=g)[:k].to(rest.device)] for _ in range(n_null)]
    d_null = torch.stack([-contrib(S) for S in draws])
    lp_null = batch_logprob_with_delta(model, L, ids, obj_ids, pos, d_null)
    null = lp_null - lp_clean
    tau = float(np.quantile(np.abs(null), 0.95))
    sig = abs(delta) > tau
    return dict(subject=subject, pred=claimed, agree=agree, conf=float(np.mean(conf)), correct=int(norm(claimed) in gold_set),
                logp_clean=lp_clean, delta=delta, tau=tau, null_mean=float(np.mean(null)), n_active=int(len(active)),
                isr=int(sig), epa=int(sig and delta < 0), top_features="|".join(map(str, top.tolist())))
