"""Stage 1: extract candidate triples by output-based prompting with paraphrase ensembles, then compute ISR and EPA."""
import argparse, os, time
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import load_sae, process_fact, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="facts.csv")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0", type=int, default=80)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n_null", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/isr.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    sae = load_sae(a.layer, a.width, a.l0, dev)
    facts = pd.read_csv(a.facts)
    if a.limit:
        facts = facts.sample(a.limit, random_state=0) if a.limit < len(facts) else facts
    done, rows, t0 = set(), [], time.time()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); rows = old.to_dict("records"); done = set(old.subject)
    for i, f in enumerate(facts.itertuples()):
        if f.subject in done:
            continue
        gold = {norm(x) for x in f.gold_aliases.split("|")}
        r = process_fact(model, tok, sae, a.layer, f.subject, gold, k=a.k, n_null=a.n_null)
        if r is None:
            print("skipped", f.subject); continue
        r.update(group=f.group, gold=f.country, cid=f.cid, sl=f.sl, layer=a.layer, k=a.k)
        rows.append(r)
        if len(rows) % 10 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{len(rows)} done, {time.time()-t0:.0f}s")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    d = pd.DataFrame(rows)
    print(d[["correct", "isr", "epa"]].mean())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
