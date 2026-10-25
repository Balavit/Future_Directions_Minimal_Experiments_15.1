"""CPU smoke test for v2 with a tiny random Gemma-2 model."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, torch, subprocess, tempfile
from transformers import Gemma2Config, Gemma2ForCausalLM
import core, core_v2

class FakeTok:
    bos_token_id, eos_token_id = 2, 1
    def encode(self, text, add_special_tokens=False): return [5 + (ord(c) % 250) for c in text]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(32 + ((int(i) - 5) % 90)) for i in ids)

torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=16, max_position_embeddings=512, sliding_window=64, pad_token_id=0, bos_token_id=2, eos_token_id=1)
model = Gemma2ForCausalLM(cfg).eval(); tok = FakeTok()
F = 256
def mk(seed):
    g = torch.Generator().manual_seed(seed)
    return core.SAE(torch.randn(64, F, generator=g) * 0.3, torch.randn(F, 64, generator=g) * 0.3, torch.randn(F, generator=g) * 0.1 - 0.05, torch.zeros(64), torch.full((F,), 0.05), "cpu")
saes = {1: mk(1), 2: mk(2)}
ids, pos, obj = core.build_ids(tok, "T1", "Paris Tower", "France")
# 1. score_objects with no delta equals seq_logprob; zero delta equals no delta
o2 = core.build_ids(tok, "T1", "Paris Tower", "Spain")[2]
lp = core_v2.score_objects(model, 0, ids, [obj, o2])
assert np.allclose(lp[0, 0], core.seq_logprob(model, ids, obj), atol=1e-3)
assert np.allclose(lp[0, 1], core.seq_logprob(model, ids, o2), atol=1e-3)
lpz = core_v2.score_objects(model, 1, ids, [obj, o2], pos, torch.zeros(3, 64))
assert lpz.shape == (3, 2) and np.allclose(lpz, lp, atol=1e-3)
lpd = core_v2.score_objects(model, 1, ids, [obj, o2], pos, torch.full((1, 64), 4.0))
assert np.abs(lpd - lp).max() > 1e-3
# 2. margin: equals log-odds
r = np.array([-1.0, -2.0, -3.0]); assert abs(core_v2.margin(r) - (-1 - np.log(np.exp(-2) + np.exp(-3)))) < 1e-9
# 3. per-fact pipeline, two layers x two ks
pool = ["Spain", "Italy", "Peru", "Japan", "Egypt", "Brazil", "Kenya", "Chile", "India", "France", "Norway", "Iran"]
rows = core_v2.process_fact_v2(model, tok, saes, [1, 2], [3, 5], "Paris Tower", {"france"}, pool, n_null=8)
assert rows and len(rows) == 4 and {(r["layer"], r["k"]) for r in rows} == {(1, 3), (1, 5), (2, 3), (2, 5)}
assert all(r["epa"] <= r["isr"] and r["n_null"] >= 5 for r in rows)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[0].items() if k not in ("competitors", "top_features")})
# 4. analysis on synthetic data (signal present) runs end to end
rng = np.random.default_rng(0); n = 400
cor = rng.integers(0, 2, n); conf = np.clip(0.3 + 0.4 * cor + rng.normal(0, .15, n), .01, .99)
recs = []
for i in range(n):
    for k in (5, 10):
        dm = -(1 + 1.5 * cor[i] + rng.normal(0, 1)); tau = 1.0
        recs.append(dict(subject=f"s{i}", correct=int(cor[i]), conf=conf[i], agree=rng.choice([1/3, 2/3, 1]), layer=12, k=k, delta_m=dm, tau_m=tau,
                         delta_lp=dm, isr=int(abs(dm) > tau), epa=int(dm < -tau), group=["head", "mid", "tail"][i % 3], split="dev" if i % 5 == 0 else "test"))
td = tempfile.mkdtemp(); pd.DataFrame(recs).to_csv(f"{td}/isr_v2.csv", index=False)
out = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "..", "analyze_v2.py"), "--isr", f"{td}/isr_v2.csv", "--out", td], capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
assert out.returncode == 0, out.stderr[-2000:]
import json; S = json.load(open(f"{td}/summary_v2.json"))
assert S["test"]["auroc"]["margin ISR support"]["value"] > 0.6 and S["test"]["lr_test_added_value_of_support"]["p"] < 0.05
assert os.path.exists(f"{td}/pilot_v2_results.png")
print("analysis ok; chosen", S["chosen_config_from_dev"], "AUROC", round(S["test"]["auroc"]["margin ISR support"]["value"], 3))
print("ALL V2 SMOKE TESTS PASSED")
