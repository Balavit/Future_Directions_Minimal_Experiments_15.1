# v3 pilot: multi-layer necessity + sufficiency, strict null, confidence-matched analysis

Copy this folder's files **into your existing pilot folder** (same venv; v1 and v2 results are not touched).
Read `ANALYSIS_PLAN_V3.md` first. Do not change settings after you have seen the test results.

```powershell
# 0. Code check on CPU (about 1-2 minutes; no GPU or download needed)
python tests\smoke_test_v3.py

# 1. New held-out facts (internet needed; excludes the v1 and v2 facts)
python make_facts_v3.py --exclude facts.csv facts_v2.csv --out facts_v3.csv

# 2. Trial: 15 facts. Note the time per fact and check there are no errors
python run_isr_v3.py --facts facts_v3.csv --model unsloth/gemma-2-2b --layers 8,12,16 --ks 5,10 --limit 15 --out results\trial_v3.csv

# 3. Full run (800 facts). Resumes if interrupted: just run the same command again
python run_isr_v3.py --facts facts_v3.csv --model unsloth/gemma-2-2b --layers 8,12,16 --ks 5,10 --out results\isr_v3.csv

# 4. Analysis
python analyze_v3.py --isr results\isr_v3.csv --out results
```

The first run downloads two more SAE files (layers 8 and 16, a few hundred MB each).

**Time.** The attribution cost is unchanged from v2 (one pass per null country gives all layers), but each fact now has about 4x more scoring passes. My estimate is 10 to 20 seconds per fact on the RTX 4070, so roughly 3 to 5 hours for 800 facts. Use the trial to check this. If it is too slow, you can stop early and analyse what exists (the dev/test split is by hash, so any prefix is valid), or use `--n_null 20 --n_corr 1` (do not go below `--n_null 20`).

**Out of memory?** Close other programs, then try `--m_comp 3`. Do not lower the number of layers, since that changes the design.

Send back: `results\summary_v3.json`, `results\isr_v3.csv`, `results\pilot_v3_results.png`, and the console line that reports the trial time per fact.
