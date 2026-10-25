"""Build the v2 reference set: larger, stratified by popularity, split into dev / test, excluding the v1 facts."""
import argparse, zlib, pandas as pd
from make_facts import sparql, CLASSES
from core import norm, FEWSHOT_SUBJECTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="facts.csv", help="v1 facts file whose items are excluded (held-out design)")
    ap.add_argument("--n_head", type=int, default=100)
    ap.add_argument("--n_mid", type=int, default=300)
    ap.add_argument("--n_tail", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="facts_v2.csv")
    a = ap.parse_args()
    rows = []
    for c in CLASSES:
        q = f"""SELECT ?item ?itemLabel ?country ?countryLabel ?sl WHERE {{
          ?item wdt:P31 wd:{c} ; wdt:P17 ?country ; wikibase:sitelinks ?sl . FILTER(?sl > 15)
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} ORDER BY DESC(?sl) LIMIT 2500"""
        for b in sparql(q):
            rows.append(dict(item=b["item"]["value"].split("/")[-1], subject=b["itemLabel"]["value"], cid=b["country"]["value"].split("/")[-1],
                             country=b["countryLabel"]["value"], sl=int(b["sl"]["value"])))
        print(c, len(rows))
    df = pd.DataFrame(rows).drop_duplicates()
    n_c = df.groupby("item").cid.nunique()
    df = df[df.item.isin(n_c[n_c == 1].index)].drop_duplicates("item")
    df = df[~df.subject.str.match(r"^Q\\d+$") & (df.subject.str.len() < 40)]
    df = df[~df.subject.duplicated(keep=False)]
    df = df[~df.apply(lambda r: norm(r.country) in norm(r.subject), axis=1)]
    df = df[~df.subject.isin(FEWSHOT_SUBJECTS)]
    try:
        old = pd.read_csv(a.exclude)
        df = df[~df.item.isin(set(old.item)) & ~df.subject.isin(set(old.subject))]
        print(f"excluded {len(old)} v1 facts")
    except FileNotFoundError:
        print("no v1 facts file found; nothing excluded")
    bins = {"head": df[df.sl >= 100], "mid": df[(df.sl >= 41) & (df.sl < 100)], "tail": df[(df.sl >= 16) & (df.sl <= 40)]}
    n = {"head": a.n_head, "mid": a.n_mid, "tail": a.n_tail}
    sel = pd.concat([b.sample(min(n[k], len(b)), random_state=a.seed).assign(group=k) for k, b in bins.items()])
    sel["split"] = sel.item.map(lambda x: "dev" if zlib.crc32(x.encode()) % 5 == 0 else "test")
    al = {}
    ids = sorted(sel.cid.unique())
    for i in range(0, len(ids), 80):
        vals = " ".join(f"wd:{x}" for x in ids[i:i + 80])
        q = f'SELECT ?c ?a WHERE {{ VALUES ?c {{ {vals} }} ?c skos:altLabel ?a . FILTER(lang(?a)="en") }}'
        for b in sparql(q):
            al.setdefault(b["c"]["value"].split("/")[-1], set()).add(b["a"]["value"])
    sel["gold_aliases"] = sel.apply(lambda r: "|".join(sorted({r.country} | al.get(r.cid, set()))), axis=1)
    sel.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sel)} facts")
    print(sel.groupby(["group", "split"]).size())


if __name__ == "__main__":
    main()
