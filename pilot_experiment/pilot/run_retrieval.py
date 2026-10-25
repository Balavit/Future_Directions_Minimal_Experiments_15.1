"""Stage 2: compare no retrieval, retrieval with all extracted triples, and retrieval with triples above the null threshold."""
import argparse, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retrieval import run_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isr", default="results/isr.csv")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--out", default="results/retrieval.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.isr)
    rng = np.random.default_rng(0)
    recs = d.to_dict("records")
    same, diff = [], []
    for _ in range(200000):
        i, j = rng.integers(0, len(recs), 2)
        if i == j:
            continue
        (same if recs[i]["cid"] == recs[j]["cid"] else diff).append((i, j))
        if len(same) >= a.pairs // 2 and len(diff) >= a.pairs // 2:
            break
    pairs = same[:a.pairs // 2] + diff[:a.pairs // 2]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map=dev).eval()
    rows = []
    for i, j in pairs:
        gx, gy = recs[i], recs[j]
        truth = "yes" if gx["cid"] == gy["cid"] else "no"
        for mode in ("none", "all", "filtered"):
            ans = run_pair(model, tok, gx, gy, mode)
            rows.append(dict(x=gx["subject"], y=gy["subject"], mode=mode, truth=truth, answer=ans,
                             wrong=int(ans in ("yes", "no") and ans != truth), abstain=int(ans == "unknown"), correct=int(ans == truth),
                             x_ok=gx["correct"], y_ok=gy["correct"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(pd.DataFrame(rows).groupby("mode")[["correct", "wrong", "abstain"]].mean())


if __name__ == "__main__":
    main()
