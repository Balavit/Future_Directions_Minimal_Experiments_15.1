# v3 pilot: multi-layer necessity + sufficiency, strict null, confidence-matched analysis

Read `ANALYSIS\_PLAN\_V3.md` first. Do not change settings after you have seen the test results.

```powershell
# 0. Code check on CPU (about 1-2 minutes; no GPU or download needed)
python tests\\smoke\_test\_v3.py

# 1. New held-out facts (internet needed; excludes the v1 and v2 facts)
python make\_facts\_v3.py --exclude facts.csv facts\_v2.csv --out facts\_v3.csv

# 2. Trial: 15 facts. Note the time per fact and check there are no errors
python run\_isr\_v3.py --facts facts\_v3.csv --model unsloth/gemma-2-2b --layers 8,12,16 --ks 5,10 --limit 15 --out results\\trial\_v3.csv

# 3. Full run (800 facts). Resumes if interrupted: just run the same command again
python run\_isr\_v3.py --facts facts\_v3.csv --model unsloth/gemma-2-2b --layers 8,12,16 --ks 5,10 --out results\\isr\_v3.csv

# 4. Analysis
python analyze\_v3.py --isr results\\isr\_v3.csv --out results

