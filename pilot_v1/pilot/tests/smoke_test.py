"""CPU smoke test with a tiny random Gemma-2 model: checks hooks, attribution, ablation, ISR/EPA, retrieval and statistics."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch
from transformers import Gemma2Config, Gemma2ForCausalLM
import core, retrieval, stats

class FakeTok:
    bos_token_id, eos_token_id = 2, 1
    def encode(self, text, add_special_tokens=False): return [5 + (ord(c) % 250) for c in text]
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(32 + ((int(i) - 5) % 90)) for i in ids)

torch.manual_seed(0)
cfg = Gemma2Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                   head_dim=16, max_position_embeddings=512, sliding_window=64, pad_token_id=0, bos_token_id=2, eos_token_id=1)
model = Gemma2ForCausalLM(cfg).eval()
tok = FakeTok()
F = 256
g = torch.Generator().manual_seed(1)
sae = core.SAE(torch.randn(64, F, generator=g) * 0.3, torch.randn(F, 64, generator=g) * 0.3, torch.randn(F, generator=g) * 0.1 - 0.05,
               torch.zeros(64), torch.full((F,), 0.05), "cpu")

# 1. hook with zero delta must not change the log-prob
b = core.build_ids(tok, "T1", "Paris Tower", "France")
ids, pos, obj = b
lp0 = core.seq_logprob(model, ids, obj)
lp1 = core.batch_logprob_with_delta(model, 1, ids, obj, pos, torch.zeros(3, 64))
assert np.allclose(lp1, lp0, atol=1e-3), (lp0, lp1)
# 2. a big delta must change it
lp2 = core.batch_logprob_with_delta(model, 1, ids, obj, pos, torch.full((1, 64), 5.0))
assert abs(lp2[0] - lp0) > 1e-3
# 3. attribution: first-order prediction of a small ablation should match the real change in sign and rough size
acts, attr, lp = core.attribution(model, 1, sae, ids, obj, pos)
assert abs(lp - lp0) < 1e-3
active = torch.nonzero(acts > 0).flatten()
assert len(active) > 10
S = active[torch.topk(attr[active], 5).indices]
eps = 0.05
delta = -(eps * acts[S].unsqueeze(1) * sae.W_dec[S]).sum(0)
real = core.batch_logprob_with_delta(model, 1, ids, obj, pos, delta.unsqueeze(0))[0] - lp0
pred = -eps * attr[S].sum().item()
print("attribution check: predicted", round(pred, 5), "real", round(float(real), 5))
assert np.sign(pred) == np.sign(real) or abs(real) < 1e-4
# 4. full per-fact pipeline
r = core.process_fact(model, tok, sae, 1, "Paris Tower", {"france"}, k=5, n_null=20)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
assert r is not None and set(["isr", "epa", "delta", "tau", "conf"]) <= set(r)
assert r["epa"] <= r["isr"]
# 5. retrieval prompts and answers
gx = dict(subject="A Tower", pred="France", isr=1, cid="Q1"); gy = dict(subject="B Bridge", pred="Spain", isr=0, cid="Q2")
for m in ("none", "all", "filtered"):
    print(m, retrieval.run_pair(model, tok, gx, gy, m))
# 6. statistics
rng = np.random.default_rng(0)
lab = rng.integers(0, 2, 200); sc = lab + rng.normal(0, 1, 200)
a = stats.auroc(sc, lab); assert 0.7 < a < 0.95
assert abs(stats.auroc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9
assert stats.mcnemar_exact([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) <= 0.25
t, p = stats.fisher([1, 1, 0, 0] * 10, [1, 1, 0, 0] * 10); assert p < 1e-6
lo, hi = stats.bootstrap(stats.auroc, sc, lab, n=200); assert lo < a < hi
print("ALL SMOKE TESTS PASSED")
