"""Build the curated reference set for the relation 'located in' (landmark -> country) from Wikidata."""
import argparse, time, requests, pandas as pd, numpy as np
from core import norm, FEWSHOT_SUBJECTS

URL = "https://query.wikidata.org/sparql"
HEAD = {"User-Agent": "minimal-kg-experiment/0.1 (research use)", "Accept": "application/sparql-results+json"}
CLASSES = ["Q570116", "Q33506", "Q16560", "Q23413", "Q39715", "Q12280", "Q11303", "Q44539", "Q16970", "Q24354", "Q483110"]


def sparql(q):
    for a in range(4):
        r = requests.get(URL, params={"query": q}, headers=HEAD, timeout=120)
        if r.status_code == 200:
            return r.json()["results"]["bindings"]
        time.sleep(5 * (a + 1))
    raise RuntimeError(f"SPARQL failed: {r.status_code} {r.text[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_head", type=int, default=120)
    ap.add_argument("--n_tail", type=int, default=120)
    ap.add_argument("--out", default="facts.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for c in CLASSES:
        q = f"""SELECT ?item ?itemLabel ?country ?countryLabel ?sl WHERE {{
          ?item wdt:P31 wd:{c} ; wdt:P17 ?country ; wikibase:sitelinks ?sl . FILTER(?sl > 15)
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} ORDER BY DESC(?sl) LIMIT 1500"""
        for b in sparql(q):
            rows.append(dict(item=b["item"]["value"].split("/")[-1], subject=b["itemLabel"]["value"], cid=b["country"]["value"].split("/")[-1],
                             country=b["countryLabel"]["value"], sl=int(b["sl"]["value"])))
        print(c, len(rows))
    df = pd.DataFrame(rows).drop_duplicates()
    n_c = df.groupby("item").cid.nunique()
    df = df[df.item.isin(n_c[n_c == 1].index)].drop_duplicates("item")
    df = df[~df.subject.str.match(r"^Q\d+$") & (df.subject.str.len() < 40)]
    df = df[~df.subject.duplicated(keep=False)]
    df = df[~df.apply(lambda r: norm(r.country) in norm(r.subject), axis=1)]
    df = df[~df.subject.isin(FEWSHOT_SUBJECTS)].sort_values("sl", ascending=False)
    head = df.head(a.n_head).assign(group="head")
    tail = df[(df.sl >= 16) & (df.sl <= 40)].sample(a.n_tail, random_state=a.seed).assign(group="tail")
    sel = pd.concat([head, tail]).drop_duplicates("item")
    al = {}
    ids = sorted(sel.cid.unique())
    for i in range(0, len(ids), 80):
        vals = " ".join(f"wd:{x}" for x in ids[i:i + 80])
        q = f'SELECT ?c ?a WHERE {{ VALUES ?c {{ {vals} }} ?c skos:altLabel ?a . FILTER(lang(?a)="en") }}'
        for b in sparql(q):
            al.setdefault(b["c"]["value"].split("/")[-1], set()).add(b["a"]["value"])
    sel["gold_aliases"] = sel.apply(lambda r: "|".join(sorted({r.country} | al.get(r.cid, set()))), axis=1)
    sel.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(sel)} facts ({(sel.group=='head').sum()} head, {(sel.group=='tail').sum()} tail)")


if __name__ == "__main__":
    main()
