"""Plain numpy / scipy statistics used by the analysis (unit-tested without a GPU)."""
import numpy as np
from scipy import stats as st


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = st.rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap(fn, *arrays, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    N = len(arrays[0])
    vals = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        v = fn(*[a[i] for a in arrays])
        if not np.isnan(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_exact(a_wrong, b_wrong):
    a, b = np.asarray(a_wrong, bool), np.asarray(b_wrong, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * st.binom.cdf(min(n01, n10), n, 0.5)))


def fisher(isr, correct):
    isr, correct = np.asarray(isr, int), np.asarray(correct, int)
    t = [[int(((isr == 1) & (correct == 1)).sum()), int(((isr == 1) & (correct == 0)).sum())],
         [int(((isr == 0) & (correct == 1)).sum()), int(((isr == 0) & (correct == 0)).sum())]]
    return t, float(st.fisher_exact(t)[1])
