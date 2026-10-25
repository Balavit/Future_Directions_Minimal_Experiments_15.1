"""CPU smoke test for v3 with a tiny random Gemma-2 model."""
import sys, os, json, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, torch
from transformers import Gemma2Config, Gemma2ForCausalLM
import core, core_v3

class FakeTok:
    bos_token_id, eos_token_id = 2, 1
    def encode(self, text, add_special_tokens=False): return [5 + (ord(c) % 250) for c in text]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(32 + ((int(i) - 5) % 90)) for i in ids)

torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=5, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=16, max_position_embeddings=512, sliding_window=64, pad_token_id=0, bos_token_id=2, eos_token_id=1)
model = Gemma2ForCausalLM(cfg).eval(); tok = FakeTok(); F = 256
def mk(seed):
    g = torch.Generator().manual_seed(seed)
    return core.SAE(torch.randn(64, F, generator=g) * 0.3, torch.randn(F, 64, generator=g) * 0.3, torch.randn(F, generator=g) * 0.1 - 0.05, torch.zeros(64), torch.full((F,), 0.05), "cpu")
layers = [1, 3]; saes = {1: mk(1), 3: mk(3)}
ids, pos, obj = core.build_ids(tok, "T1", "Paris Tower", "France")
o2 = core.build_ids(tok, "T1", "Paris Tower", "Spain")[2]
# 1. scoring without / with zero deltas matches plain log-probs; a real delta changes them
lp = core_v3.score_multi(model, ids, [obj, o2])
assert np.allclose(lp[0, 0], core.seq_logprob(model, ids, obj), atol=1e-3)
z = {L: torch.zeros(3, 64) for L in layers}
assert np.allclose(core_v3.score_multi(model, ids, [obj, o2], pos, z), lp, atol=1e-3)
assert np.abs(core_v3.score_multi(model, ids, [obj, o2], pos, {1: torch.full((1, 64), 4.0), 3: torch.full((1, 64), 4.0)}) - lp).max() > 1e-3
# 2. multi-layer attribution equals the single-layer routine at every layer
att, lpm = core_v3.attribution_multi(model, layers, saes, ids, obj, pos)
for L in layers:
    a1, at1, lp1 = core.attribution(model, L, saes[L], ids, obj, pos)
    assert np.allclose(att[L][1].numpy(), at1.numpy(), atol=1e-3, rtol=1e-3), L
    assert np.allclose(att[L][0].numpy(), a1.numpy(), atol=1e-4)
# 3. capture_acts matches the activations seen by the attribution pass
ca = core_v3.capture_acts(model, layers, saes, ids, pos)
assert all(np.allclose(ca[L].numpy(), att[L][0].numpy(), atol=1e-4) for L in layers)
# 4. sufficiency identity: writing the same values into the same prompt changes nothing
S = torch.topk(att[1][0], 5).indices
d0 = {L: ((att[L][0][S] - ca[L][S]).unsqueeze(1) * saes[L].W_dec[S]).sum(0, keepdim=True) for L in layers}
assert np.allclose(core_v3.score_multi(model, ids, [obj], pos, d0), core_v3.score_multi(model, ids, [obj]), atol=1e-3)
# 5. full per-fact pipeline
pool = ["Spain", "Italy", "Peru", "Japan", "Egypt", "Brazil", "Kenya", "Chile", "India", "France", "Norway", "Iran", "Cuba", "Chad"]
cp = [("Rome Tower", "Italy"), ("Lima Hall", "Peru"), ("Tokyo Gate", "Japan"), ("Cairo Mosque", "Egypt")]
core_v3.greedy_text = lambda *a, **k: "Spain"   # the random model cannot generate a country; fix the claim
rows = core_v3.process_fact_v3(model, tok, saes, layers, [3, 5], "Paris Tower", {"france"}, pool, cp, n_null=10, n_corr=2)
assert rows and len(rows) == 2 and all(r["n_null"] >= 10 and r["n_corr"] == 2 for r in rows)
assert all(0 < r["nec_p"] <= 1 and 0 < r["suf_p"] <= 1 for r in rows)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[0].items() if k not in ("competitors",)})
# 6. analysis end to end on synthetic data with signal
rng = np.random.default_rng(0); n = 500; cor = rng.integers(0, 2, n); conf = np.clip(.3 + .4 * cor + rng.normal(0, .15, n), .01, .99); recs = []
for i in range(n):
    for k in (5, 10):
        nz, sz = rng.normal(2 * cor[i] - 0.5, 1), rng.normal(1.5 * cor[i] - 0.5, 1)
        recs.append(dict(subject=f"s{i}", correct=int(cor[i]), conf=conf[i], agree=rng.choice([1/3, 2/3, 1]), k=k, nec_z=nz, suf_z=sz, isr3=int(nz > 1.5 and sz > 1.5),
                         group=["head", "mid", "tail"][i % 3], split="dev" if i % 5 == 0 else "test"))
td = tempfile.mkdtemp(); pd.DataFrame(recs).to_csv(f"{td}/isr_v3.csv", index=False)
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "analyze_v3.py"), "--isr", f"{td}/isr_v3.csv", "--out", td],
                     capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
assert out.returncode == 0, out.stderr[-2000:]
S = json.load(open(f"{td}/summary_v3.json")); assert S["verdict"]["supported"] and os.path.exists(f"{td}/pilot_v3_results.png")
print("analysis ok:", S["verdict"], "matched AUROC", round(S["test"]["H3_confidence_matched_auroc"]["value"], 3))
# and on pure noise it must NOT report support
for r in recs: r["nec_z"], r["suf_z"] = rng.normal(0, 1), rng.normal(0, 1)
pd.DataFrame(recs).to_csv(f"{td}/isr_v3.csv", index=False)
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "analyze_v3.py"), "--isr", f"{td}/isr_v3.csv", "--out", td], capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
assert out.returncode == 0, out.stderr[-2000:]
assert not json.load(open(f"{td}/summary_v3.json"))["verdict"]["supported"]
print("noise control ok")
print("ALL V3 SMOKE TESTS PASSED")
