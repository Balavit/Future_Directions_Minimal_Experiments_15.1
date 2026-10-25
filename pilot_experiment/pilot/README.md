# Minimal experiment: "located in" (landmark -> country)

Tests the core of the thesis on one open model: are the edges extracted by output-based prompting
causally supported by the model's own sparse-autoencoder features (ISR / EPA), and does filtering by
that support lower the hallucination rate of a downstream graph-retrieval task (hypothesis H4)?

* Model: `google/gemma-2-2b` (bf16, about 5.5 GB)   \* SAE: Gemma Scope residual SAE, layer 12, width 16k
* Relation: "located in" (landmark -> country), 120 well-known + 120 less-known landmarks from Wikidata

## 1\. Install (Windows 11, PowerShell)

1. Install Python 3.11 from python.org (tick "Add python.exe to PATH"). Update the NVIDIA driver (GeForce Experience or nvidia.com). Check with `nvidia-smi`.
2. In the folder that contains this README:

```
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is\_available(), torch.cuda.get\_device\_name(0))"
```

The last line must print `True` and the GPU name. If it prints False, reinstall torch with the command above.

## 2\. Get access to Gemma (one time)

1. Create a free account at huggingface.co. Open https://huggingface.co/google/gemma-2-2b and accept the licence.
2. Create a read token (Settings > Access Tokens) and log in: `huggingface-cli login` (or `hf auth login`) and paste the token.

## 3\. Check the code (about 1 minute, CPU only)

```
python tests\\smoke\_test.py
```

It must end with `ALL SMOKE TESTS PASSED`.

## 4\. Build the reference facts (needs internet, 1-3 minutes)

```
python make\_facts.py
```

This writes `facts.csv` (about 240 facts). Open it and look at 20 rows to make sure subjects and countries are sensible.

## 5\. Run the experiment

Quick trial first (about 5 minutes):

```
python run\_isr.py --limit 20
```

Then the full run (about 20-45 minutes; it saves progress and can be resumed by running the same command again):

```
python run\_isr.py
python run\_retrieval.py
python analyze.py
```

Outputs are in `results\\`: `isr.csv`, `retrieval.csv`, `summary.json`, `pilot\_results.png`.

## Optional robustness checks (run after the main result, each is a new full run)

```
python run\_isr.py --layer 8  --out results\\isr\_L8.csv
python run\_isr.py --layer 16 --out results\\isr\_L16.csv
python run\_isr.py --k 3 --out results\\isr\_k3.csv
```

## How the measures are computed

* Extraction: three paraphrased few-shot prompts; the majority answer is the claimed object. Correct if it matches a Wikidata country label or alias.
* Cited features: the k=5 SAE features at the last subject token with the largest attribution (activation x gradient) to the object's log-probability.
* Delta: change in log-probability of the claimed object after removing those features' decoder contribution at that position.
* Null threshold tau (selection-matched): 95th percentile of |Delta| when ablating the features that would be cited for 20 OTHER countries at the same position. This avoids the bias of comparing selected features with random ones.
* ISR: |Delta| > tau. EPA: |Delta| > tau and Delta < 0 (ablation lowers support, as the graph predicts).
* Downstream task: "Are X and Y located in the same country?" with no context, with all extracted triples, or only triples with ISR = 1. Hallucination = a wrong yes/no answer; "unknown" counts as abstention.

## 

