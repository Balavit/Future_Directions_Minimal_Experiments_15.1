"""Stage 3: statistics and figures. Writes results/summary.json (send this file back) and PNG figures."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from stats import auroc, bootstrap, mcnemar_exact, fisher

INK, BLUE, TEAL, SAND, ROSE, GRAY = "#1B2A41", "#2F5D8A", "#2C8C8C", "#C79A42", "#A8473F", "#5E6672"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isr", default="results/isr.csv")
    ap.add_argument("--retrieval", default="results/retrieval.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    d = pd.read_csv(a.isr)
    S = {"n_facts": int(len(d))}
    S["accuracy"] = float(d.correct.mean())
    S["accuracy_by_group"] = d.groupby("group").correct.mean().round(3).to_dict()
    S["isr_rate"] = float(d.isr.mean()); S["epa_rate"] = float(d.epa.mean())
    S["isr_rate_by_correct"] = d.groupby("correct").isr.mean().round(3).to_dict()
    t, p = fisher(d.isr, d.correct)
    S["fisher_table[isr x correct]"] = t; S["fisher_p"] = p
    d["support"] = -d.delta / d.tau
    scores = {"ISR support (-delta/tau)": d.support.values, "model confidence": d.conf.values, "template agreement": d.agree.values}
    S["auroc"] = {}
    if d.correct.nunique() == 2:
        for k, v in scores.items():
            S["auroc"][k] = dict(value=auroc(v, d.correct), ci95=bootstrap(auroc, v, d.correct.values))
        X = np.c_[d.support.clip(-20, 20), np.log(d.conf.clip(1e-6)), d.agree]
        pr = cross_val_predict(LogisticRegression(max_iter=1000), X, d.correct, cv=5, method="predict_proba")[:, 1]
        S["auroc"]["combined (5-fold CV)"] = dict(value=auroc(pr, d.correct), ci95=bootstrap(auroc, pr, d.correct.values))
    else:
        S["note"] = "Only one correctness class present; AUROC undefined."
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    c = np.where(d.correct == 1, TEAL, ROSE)
    ax[0].scatter(d.tau, -d.delta, c=c, s=14, alpha=0.8, lw=0)
    lim = max(d.tau.max(), (-d.delta).max()) * 1.05
    ax[0].plot([0, lim], [0, lim], color=GRAY, lw=0.8, ls="--")
    ax[0].set(xlabel="null threshold τ", ylabel="drop in log-prob (−Δ)", title="(a) Ablation effect vs null")
    for lab, col in (("correct", TEAL), ("incorrect", ROSE)):
        ax[0].scatter([], [], c=col, s=14, label=lab)
    ax[0].legend(frameon=False, fontsize=7)
    if d.correct.nunique() == 2:
        for k, col in zip(["ISR support (-delta/tau)", "model confidence", "template agreement"], [BLUE, SAND, GRAY]):
            fpr, tpr, _ = roc_curve(d.correct, scores[k])
            ax[1].plot(fpr, tpr, color=col, lw=1.4, label=f"{k.split(' (')[0]} ({S['auroc'][k]['value']:.2f})")
        ax[1].plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
        ax[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="(b) Predicting correctness")
    if os.path.exists(a.retrieval):
        r = pd.read_csv(a.retrieval)
        S["retrieval"] = {}
        names = {"none": "No retrieval", "all": "All triples", "filtered": "Above null"}
        xs = []
        for m in ["none", "all", "filtered"]:
            g = r[r["mode"] == m]
            S["retrieval"][m] = dict(n=int(len(g)), accuracy=float(g.correct.mean()), hallucination=float(g.wrong.mean()), abstention=float(g.abstain.mean()),
                                     hallucination_ci95=bootstrap(np.mean, g.wrong.values))
            xs.append(S["retrieval"][m]["hallucination"])
        piv = r.pivot_table(index=["x", "y"], columns="mode", values="wrong")
        S["mcnemar_p"] = {"all_vs_filtered": mcnemar_exact(piv["all"], piv["filtered"]), "none_vs_filtered": mcnemar_exact(piv["none"], piv["filtered"])}
        ax[2].bar(range(3), xs, color=[GRAY, ROSE, TEAL])
        ax[2].set_xticks(range(3)); ax[2].set_xticklabels([names[m] for m in ["none", "all", "filtered"]], fontsize=7)
        ax[2].set(ylabel="hallucination rate", title="(c) Downstream task")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{a.out}/pilot_results.png", dpi=300)
    json.dump(S, open(f"{a.out}/summary.json", "w"), indent=2, default=float)
    print(json.dumps(S, indent=2, default=float))


if __name__ == "__main__":
    main()
