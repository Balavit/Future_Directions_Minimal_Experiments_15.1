# v2 pilot: margin effect, confidence control, k/layer sweep, held-out split

Run from PowerShell inside the folder, with the venv active:

```powershell
# 0. CPU check of the code (about 1 minute)
python tests\\smoke\_test\_v2.py

# 1. New, held-out facts (needs internet; excludes your v1 facts.csv)
python make\_facts\_v2.py --exclude facts.csv --out facts\_v2.csv

# 2. Quick trial (20 facts, a few minutes) 
python run\_isr\_v2.py --facts facts\_v2.csv --model unsloth/gemma-2-2b --layers 12 --ks 5,10 --limit 20 --out results\\trial\_v2.csv

# 3. Full first v2: one layer, two k values, 800 facts (expect \~1.5-3 h on the RTX 4070; it resumes if interrupted)
python run\_isr\_v2.py --facts facts\_v2.csv --model unsloth/gemma-2-2b --layers 12 --ks 5,10 --out results\\isr\_v2.csv

# 4. Analysis (reads dev/test split, writes the summary and figure)
python analyze\_v2.py --isr results\\isr\_v2.csv --out results
```



