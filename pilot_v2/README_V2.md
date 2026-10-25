# v2 pilot: margin effect, confidence control, k/layer sweep, held-out split

Copy the contents of this folder **into your existing pilot folder** (same venv, same `core.py`, `stats.py`, `make_facts.py`). Nothing from v1 is overwritten, and v1 results stay in `results\`.

Run from PowerShell inside the folder, with the venv active:

```powershell
# 0. CPU check of the code (about 1 minute)
python tests\smoke_test_v2.py

# 1. New, held-out facts (needs internet; excludes your v1 facts.csv)
python make_facts_v2.py --exclude facts.csv --out facts_v2.csv

# 2. Quick trial (20 facts, a few minutes) 
python run_isr_v2.py --facts facts_v2.csv --model unsloth/gemma-2-2b --layers 12 --ks 5,10 --limit 20 --out results\trial_v2.csv

# 3. Full first v2: one layer, two k values, 800 facts (expect ~1.5-3 h on the RTX 4070; it resumes if interrupted)
python run_isr_v2.py --facts facts_v2.csv --model unsloth/gemma-2-2b --layers 12 --ks 5,10 --out results\isr_v2.csv

# 4. Analysis (reads dev/test split, writes the summary and figure)
python analyze_v2.py --isr results\isr_v2.csv --out results
```

Send back `results\summary_v2.json`, `results\isr_v2.csv` and `results\pilot_v2_results.png`.

If a trial run exceeds GPU memory, nothing else needs changing: lower the load with `--m_comp 3 --n_null 12`.
Later extension (second run, not now): `--layers 6,12,18` (SAEs are downloaded automatically) to get the layer sweep. Run it only after the first v2 is analysed, and report it as a separate, labelled sweep.

Read `ANALYSIS_PLAN.md` first. Do not change settings after looking at the test results.
