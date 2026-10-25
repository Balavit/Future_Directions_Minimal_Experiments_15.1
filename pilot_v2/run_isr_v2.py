"""v2 stage 1: extraction + margin-based ISR/EPA for several layers and numbers of cited features."""
import argparse, os, time
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import load_sae, norm
from core_v2 import process_fact_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="facts_v2.csv")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layers", default="12")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0", type=int, default=80)
    ap.add_argument("--n_null", type=int, default=20)
    ap.add_argument("--m_comp", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/isr_v2.csv")
    a = ap.parse_args()
    layers, ks = [int(x) for x in a.layers.split(",")], [int(x) for x in a.ks.split(",")]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    saes = {L: load_sae(L, a.width, a.l0, dev) for L in layers}
    facts = pd.read_csv(a.facts)
    pool = sorted(facts.country.unique().tolist())
    if a.limit:
        facts = facts.sample(a.limit, random_state=0) if a.limit < len(facts) else facts
    rows, done, t0 = [], set(), time.time()
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); rows = old.to_dict("records"); done = set(old.subject)
    nf = 0
    for f in facts.itertuples():
        if f.subject in done:
            continue
        gold = {norm(x) for x in f.gold_aliases.split("|")}
        rs = process_fact_v2(model, tok, saes, layers, ks, f.subject, gold, pool, n_null=a.n_null, m_comp=a.m_comp)
        if rs is None:
            print("skipped", f.subject); continue
        for r in rs:
            r.update(group=f.group, split=getattr(f, "split", "test"), gold=f.country, cid=f.cid, sl=f.sl)
        rows += rs
        nf += 1
        if nf % 10 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{nf} facts done, {time.time() - t0:.0f}s")
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(d.groupby(["layer", "k"])[["correct", "isr", "epa"]].mean())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
