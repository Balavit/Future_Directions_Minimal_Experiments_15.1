"""v3 stage: multi-layer necessity + sufficiency support (ISR3) for each fact."""
import argparse, os, time
import pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core import load_sae, norm
from core_v3 import process_fact_v3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="facts_v3.csv")
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--layers", default="8,12,16")
    ap.add_argument("--ks", default="5,10")
    ap.add_argument("--width", default="16k")
    ap.add_argument("--l0", type=int, default=80)
    ap.add_argument("--n_null", type=int, default=30)
    ap.add_argument("--n_corr", type=int, default=2)
    ap.add_argument("--m_comp", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/isr_v3.csv")
    a = ap.parse_args()
    assert a.n_null >= 20, "n_null must be >= 20 so that p <= 0.05 is reachable"
    layers, ks = [int(x) for x in a.layers.split(",")], [int(x) for x in a.ks.split(",")]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map=dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    saes = {L: load_sae(L, a.width, a.l0, dev) for L in layers}
    facts = pd.read_csv(a.facts)
    pool = sorted(facts.country.unique().tolist())
    corr_pool = list(zip(facts.subject, facts.country))
    if a.limit and a.limit < len(facts):
        facts = facts.sample(a.limit, random_state=0)
    rows, done, t0, nf = [], set(), time.time(), 0
    if os.path.exists(a.out):
        old = pd.read_csv(a.out); rows = old.to_dict("records"); done = set(old.subject)
    for f in facts.itertuples():
        if f.subject in done:
            continue
        gold = {norm(x) for x in f.gold_aliases.split("|")}
        rs = process_fact_v3(model, tok, saes, layers, ks, f.subject, gold, pool, corr_pool, n_null=a.n_null, m_comp=a.m_comp, n_corr=a.n_corr)
        if rs is None:
            print("skipped", f.subject); continue
        for r in rs:
            r.update(group=f.group, split=getattr(f, "split", "test"), gold=f.country, cid=f.cid, sl=f.sl)
        rows += rs; nf += 1
        if nf % 10 == 0:
            pd.DataFrame(rows).to_csv(a.out, index=False)
            print(f"{nf} facts done, {time.time() - t0:.0f}s")
    d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
    print(d.groupby("k")[["correct", "isr3", "nec_z", "suf_z"]].mean()); print("wrote", a.out)


if __name__ == "__main__":
    main()
