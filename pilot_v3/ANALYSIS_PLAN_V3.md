# v3 analysis plan (fixed before any v3 data are generated)

**History.** v1 (240 facts; one layer; raw log-probability) and v2 (705 facts; margin effect; held-out) both found that the support measure ISR did not predict correctness (v2 test AUROC 0.48 vs 0.76 for model confidence; after adjusting for confidence the ISR effect was significant in the *wrong* direction; AUROC within the high-confidence band 0.27). v3 is a redesign motivated by those results, run on **new facts** (v1 and v2 facts excluded). Whatever v3 finds, v1, v2 and v3 are all reported.

**What changes (and why).**
1. *Multi-layer intervention.* The cited features of layers 8, 12 and 16 (top-k per layer, k chosen on dev from {5, 10}) are ablated together at the subject's last token. v2 showed that a single-layer ablation removes the same margin for correct and incorrect claims, a sign of redundancy across layers.
2. *Two tests instead of one.* **Necessity**: drop in the margin (claimed country vs the 5 strongest competitors) after ablation. **Sufficiency**: in a prompt about a *different* landmark in a different country, the same features are set to their clean-run values; support is the gain in the claimed country's margin (averaged over 2 corrupted prompts).
3. *Stricter null.* For each test, the null is the same procedure with the features cited for 30 other countries. A test passes if the claim's effect exceeds **all 30** null effects (p <= 0.05, since the smallest attainable p is 1/31). **ISR3** requires both tests to pass.
4. *Confidence control by design.* Primary inference is confidence-matched (see H3).

**Score.** s = min(necessity z, sufficiency z), each z-score relative to its own null, clipped to [-10, 10]; high s means both tests support the claim.

**Hypotheses (test split only; k fixed from dev).**
- H1: AUROC of s for correctness > 0.5; lower bound of the 95% bootstrap interval above 0.5.
- H2 (primary): likelihood-ratio test of `correct ~ log(conf) + agreement + s` against `correct ~ log(conf) + agreement` has p < 0.05 **and** a positive coefficient for s.
- H3: confidence-matched AUROC (AUROC within confidence quintiles, averaged with weights n_pos x n_neg) > 0.5, lower 95% bootstrap bound above 0.5.
- H4 (descriptive): ISR3 pass rate for correct vs incorrect claims (Fisher exact). The overall pass rate should be well below the 80% seen in v1 and v2; if it is not, the null is still too permissive.

**Verdict rule.** The faithfulness measure is called *supported as a correctness signal* only if H1, H2 and H3 all hold. Otherwise it is *not supported*. The script prints this verdict. No other settings are tried after the test split has been analysed. Anything beyond H1-H4 is labelled exploratory.

**What v3 does not test.** (a) Whether the cited features predict the effect of an *edit* of the model (EPA); that is planned as v4. (b) Models other than Gemma 2 2B, relations other than "located in", other SAE widths. (c) Truth: ISR measures whether the model's own mechanism supports the claim. A wrong claim can be strongly supported by the model's mechanism, so even a successful v3 would show only that ISR carries information about the reliability of a claim, not that it certifies truth.

**Known approximations.** Feature contributions at different layers overlap (the residual stream carries earlier contributions forward), so summing ablations across layers may over-remove; the null uses the same procedure and the same number of features per layer, so the comparison remains matched. Attribution is first-order. About 150-250 incorrect test claims are expected, which limits power for small effects.
