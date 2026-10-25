"""v2 stage 2: pre-registered analysis. Settings (layer, k) are chosen on the DEV split only; all reported tests use the TEST split.
Writes results/summary_v2.json and results/pilot_v2_results.png."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from stats import auroc, bootstrap, fisher

INK, BLUE, TEAL, SAND, ROSE, GRAY = "#1B2A41", "#2F5D8A", "#2C8C8C", "#C79A42", "#A8473F", "#5E6672"


def loglik(X, y):
    m = LogisticRegression(C=1e6, max_iter=5000).fit(X, y)
    p = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lr_test(base, full, y):
    """Likelihood-ratio test: does `full` (base + support) fit better than `base` (confidence, agreement)?"""
    stat = 2 * (loglik(full, y) - loglik(base, y)); df = full.shape[1] - base.shape[1]
    return dict(chi2=stat, df=df, p=float(chi2.sf(max(stat, 0), df)))


def feats(d):
    sup = (-d.delta_m / d.tau_m.clip(lower=1e-6)).clip(-20, 20).values
    base = np.c_[np.log(d.conf.clip(lower=1e-6)), d.agree]
    return sup, base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isr", default="results/isr_v2.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    d = pd.read_csv(a.isr)
    S = {"n_facts": int(d.subject.nunique()), "configs": sorted({(int(l), int(k)) for l, k in zip(d.layer, d.k)})}
    S["accuracy"] = float(d.drop_duplicates("subject").correct.mean())
    dev, test = d[d.split == "dev"], d[d.split == "test"]
    S["n_dev"], S["n_test"] = int(dev.subject.nunique()), int(test.subject.nunique())
    # 1. choose the setting on dev: largest gain in AUROC of [confidence + support] over confidence alone (cross-fitted not needed: one scalar per config)
    gains = {}
    for (L, k), g in dev.groupby(["layer", "k"]):
        if g.correct.nunique() < 2:
            continue
        sup, base = feats(g)
        gains[(int(L), int(k))] = auroc(sup, g.correct.values)
    S["dev_auroc_support_by_config"] = {f"L{L}_k{k}": v for (L, k), v in gains.items()}
    best = max(gains, key=gains.get) if gains else (int(d.layer.iloc[0]), int(d.k.iloc[0]))
    S["chosen_config_from_dev"] = {"layer": best[0], "k": best[1]}
    t = test[(test.layer == best[0]) & (test.k == best[1])].copy()
    y = t.correct.values
    S["test"] = {"n": int(len(t)), "accuracy": float(y.mean()), "n_incorrect": int((1 - y).sum())}
    if len(set(y)) < 2:
        S["note"] = "single correctness class in test; analysis not defined"
        json.dump(S, open(f"{a.out}/summary_v2.json", "w"), indent=2, default=float); return
    sup, base = feats(t)
    # H1: ISR support (margin-based) predicts correctness above chance
    sc = {"margin ISR support": sup, "model confidence": t.conf.values, "template agreement": t.agree.values,
          "raw log-prob drop (v1-style)": (-t.delta_lp / t.tau_m.clip(lower=1e-6)).values}
    S["test"]["auroc"] = {k: dict(value=auroc(v, y), ci95=bootstrap(auroc, v, y)) for k, v in sc.items()}
    # H2: added value over confidence (primary confirmatory test)
    S["test"]["lr_test_added_value_of_support"] = lr_test(base, np.c_[base, sup], y)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(np.c_[base, sup], y)
    S["test"]["support_coef_adjusted"] = float(m.coef_[0][-1])
    # H3: within confidence bands (terciles), AUROC of support
    t["band"] = pd.qcut(t.conf.rank(method="first"), 3, labels=["low", "mid", "high"])
    S["test"]["auroc_support_within_confidence_band"] = {
        b: (auroc(sup[(t.band == b).values], y[(t.band == b).values]) if len(set(y[(t.band == b).values])) == 2 else None) for b in ["low", "mid", "high"]}
    # H4: ISR / EPA rates by correctness, Fisher test
    S["test"]["isr_rate_by_correct"] = t.groupby("correct").isr.mean().round(3).to_dict()
    S["test"]["epa_rate_by_correct"] = t.groupby("correct").epa.mean().round(3).to_dict()
    tb, p = fisher(t.isr, t.correct); S["test"]["fisher_isr_x_correct"] = dict(table=tb, p=p)
    S["test"]["by_group"] = {g: dict(n=int(len(x)), accuracy=float(x.correct.mean()), auroc_support=(auroc((-x.delta_m / x.tau_m.clip(lower=1e-6)).values, x.correct.values) if x.correct.nunique() == 2 else None))
                             for g, x in t.groupby("group")}
    # sweep table (descriptive, on test): not used for selection
    sw = []
    for (L, k), g in test.groupby(["layer", "k"]):
        if g.correct.nunique() == 2:
            s2, b2 = feats(g)
            sw.append(dict(layer=int(L), k=int(k), auroc_support=auroc(s2, g.correct.values), isr_rate=float(g.isr.mean()),
                           lr_p=lr_test(b2, np.c_[b2, s2], g.correct.values)["p"]))
    S["test_sweep_descriptive"] = sw
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    c = np.where(y == 1, TEAL, ROSE)
    ax[0].scatter(t.tau_m, -t.delta_m, c=c, s=12, alpha=0.8, lw=0)
    lim = float(max(t.tau_m.max(), (-t.delta_m).max())) * 1.05
    ax[0].plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls="--")
    ax[0].set(xlabel="null threshold τ (margin)", ylabel="drop in margin (−Δ)", title="(a) Margin effect vs null")
    for lab, col in (("correct", TEAL), ("incorrect", ROSE)):
        ax[0].scatter([], [], c=col, s=12, label=lab)
    ax[0].legend(frameon=False, fontsize=7)
    names = list(sc)[:3]
    ax[1].bar(range(3), [S["test"]["auroc"][n]["value"] for n in names], color=[BLUE, SAND, GRAY],
              yerr=[[S["test"]["auroc"][n]["value"] - S["test"]["auroc"][n]["ci95"][0] for n in names], [S["test"]["auroc"][n]["ci95"][1] - S["test"]["auroc"][n]["value"] for n in names]],
              ecolor=INK, capsize=2)
    ax[1].axhline(0.5, color="#bbb", lw=0.8, ls="--")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["ISR", "confidence", "agreement"], fontsize=7); ax[1].set(ylabel="AUROC (test)", title="(b) Predicting correctness", ylim=(0.3, 1))
    bands = S["test"]["auroc_support_within_confidence_band"]
    ax[2].bar(range(3), [bands[b] if bands[b] is not None else 0 for b in ["low", "mid", "high"]], color=BLUE)
    ax[2].axhline(0.5, color="#bbb", lw=0.8, ls="--")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["low", "mid", "high"], fontsize=7)
    ax[2].set(xlabel="confidence band", ylabel="AUROC of ISR", title="(c) ISR within bands", ylim=(0.3, 1))
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{a.out}/pilot_v2_results.png", dpi=300)
    json.dump(S, open(f"{a.out}/summary_v2.json", "w"), indent=2, default=float)
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
