"""v3 pre-registered analysis. k is chosen on DEV; all tests use TEST. Writes results/summary_v3.json and pilot_v3_results.png."""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from stats import auroc, bootstrap, fisher

INK, BLUE, TEAL, SAND, ROSE, GRAY = "#1B2A41", "#2F5D8A", "#2C8C8C", "#C79A42", "#A8473F", "#5E6672"


def score(d):
    return np.minimum(d.nec_z.clip(-10, 10), d.suf_z.clip(-10, 10)).values       # both tests must support the claim


def loglik(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lr_test(base, full, y):
    st = 2 * (loglik(full, y) - loglik(base, y))
    return dict(chi2=st, df=full.shape[1] - base.shape[1], p=float(chi2.sf(max(st, 0), full.shape[1] - base.shape[1])))


def strat_auroc(s, y, c, bins=5):
    """AUROC within confidence quintiles, averaged with weights n_pos * n_neg (a confidence-matched AUROC)."""
    q = pd.qcut(pd.Series(c).rank(method="first"), bins, labels=False).values
    num = den = 0.0
    for b in range(bins):
        m = q == b
        npos, nneg = int(y[m].sum()), int((1 - y[m]).sum())
        if npos and nneg:
            w = npos * nneg; num += w * auroc(s[m], y[m]); den += w
    return num / den if den else float("nan")


def boot_strat(s, y, c, n=1000, seed=0):
    r = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y)); out.append(strat_auroc(s[i], y[i], c[i]))
    return [float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isr", default="results/isr_v3.csv"); ap.add_argument("--out", default="results")
    a = ap.parse_args()
    d = pd.read_csv(a.isr)
    S = {"n_facts": int(d.subject.nunique()), "accuracy": float(d.drop_duplicates("subject").correct.mean())}
    dev, test = d[d.split == "dev"], d[d.split == "test"]
    S["n_dev"], S["n_test"] = int(dev.subject.nunique()), int(test.subject.nunique())
    gains = {int(k): auroc(score(g), g.correct.values) for k, g in dev.groupby("k") if g.correct.nunique() == 2}
    S["dev_auroc_by_k"] = gains
    k = max(gains, key=gains.get) if gains else int(d.k.iloc[0])
    S["chosen_k_from_dev"] = k
    t = test[test.k == k].copy(); y = t.correct.values
    S["test"] = {"n": int(len(t)), "accuracy": float(y.mean()), "n_incorrect": int((1 - y).sum())}
    s = score(t); conf = t.conf.values
    base = np.c_[np.log(t.conf.clip(lower=1e-6)), t.agree]
    cmp = {"ISR3 score (min of necessity, sufficiency z)": s, "necessity z": t.nec_z.clip(-10, 10).values, "sufficiency z": t.suf_z.clip(-10, 10).values,
           "model confidence": conf, "template agreement": t.agree.values}
    S["test"]["auroc"] = {n: dict(value=auroc(v, y), ci95=bootstrap(auroc, v, y)) for n, v in cmp.items()}
    S["test"]["H2_lr_test_added_value"] = lr_test(base, np.c_[base, s], y)
    S["test"]["H2_support_coef"] = float(LogisticRegression(C=1e6, max_iter=5000).fit(np.c_[base, s], y).coef_[0][-1])
    S["test"]["H3_confidence_matched_auroc"] = dict(value=strat_auroc(s, y, conf), ci95=boot_strat(s, y, conf))
    t["band"] = pd.qcut(t.conf.rank(method="first"), 3, labels=["low", "mid", "high"])
    S["test"]["auroc_within_band"] = {b: (auroc(s[(t.band == b).values], y[(t.band == b).values]) if len(set(y[(t.band == b).values])) == 2 else None) for b in ["low", "mid", "high"]}
    S["test"]["isr3_rate"] = float(t.isr3.mean())
    S["test"]["isr3_rate_by_correct"] = t.groupby("correct").isr3.mean().round(3).to_dict()
    tb, p = fisher(t.isr3, t.correct); S["test"]["H4_fisher_isr3_x_correct"] = dict(table=tb, p=p)
    S["test"]["by_group"] = {g: dict(n=int(len(x)), accuracy=float(x.correct.mean())) for g, x in t.groupby("group")}
    h = S["test"]
    h1 = h["auroc"]["ISR3 score (min of necessity, sufficiency z)"]["ci95"][0] > 0.5
    h2 = h["H2_lr_test_added_value"]["p"] < 0.05 and h["H2_support_coef"] > 0
    h3 = h["H3_confidence_matched_auroc"]["ci95"][0] > 0.5
    S["verdict"] = dict(H1=bool(h1), H2=bool(h2), H3=bool(h3), supported=bool(h1 and h2 and h3))
    S["test_k_sweep_descriptive"] = [dict(k=int(kk), auroc=auroc(score(g), g.correct.values), isr3_rate=float(g.isr3.mean())) for kk, g in test.groupby("k") if g.correct.nunique() == 2]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax[0].scatter(t.nec_z[y == 1], t.suf_z[y == 1], s=9, c=TEAL, alpha=.7, lw=0, label="correct")
    ax[0].scatter(t.nec_z[y == 0], t.suf_z[y == 0], s=9, c=ROSE, alpha=.8, lw=0, label="incorrect")
    ax[0].axhline(0, color="#bbb", lw=.7); ax[0].axvline(0, color="#bbb", lw=.7)
    ax[0].set(xlabel="necessity z", ylabel="sufficiency z", title="(a) Two tests"); ax[0].legend(frameon=False, fontsize=7)
    nm = ["ISR3 score (min of necessity, sufficiency z)", "model confidence", "template agreement"]
    v = [h["auroc"][n]["value"] for n in nm]; ci = [h["auroc"][n]["ci95"] for n in nm]
    ax[1].bar(range(3), v, color=[BLUE, SAND, GRAY], yerr=[[a_ - c[0] for a_, c in zip(v, ci)], [c[1] - a_ for a_, c in zip(v, ci)]], ecolor=INK, capsize=2)
    ax[1].axhline(.5, color="#bbb", lw=.8, ls="--"); ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["ISR3", "confidence", "agreement"], fontsize=7)
    ax[1].set(ylabel="AUROC (test)", title="(b) Correctness", ylim=(.3, 1))
    m = h["H3_confidence_matched_auroc"]
    ax[2].bar([0], [m["value"]], color=BLUE, yerr=[[m["value"] - m["ci95"][0]], [m["ci95"][1] - m["value"]]], ecolor=INK, capsize=2)
    ax[2].bar(range(1, 4), [h["auroc_within_band"][b] or 0 for b in ["low", "mid", "high"]], color=TEAL)
    ax[2].axhline(.5, color="#bbb", lw=.8, ls="--"); ax[2].set_xticks(range(4)); ax[2].set_xticklabels(["matched", "low", "mid", "high"], fontsize=7)
    ax[2].set(ylabel="AUROC of ISR3", title="(c) Confidence-controlled", ylim=(.3, 1))
    for x in ax: x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.out}/pilot_v3_results.png", dpi=300)
    json.dump(S, open(f"{a.out}/summary_v3.json", "w"), indent=2, default=float); print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
