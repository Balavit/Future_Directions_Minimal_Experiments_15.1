# v2 analysis plan (written before any v2 data were generated)

**Context.** The v1 pilot (240 facts, layer 12, k = 5) found that the Internal Support Rate (ISR) did not predict correctness better than model confidence (AUROC 0.57 vs 0.79; ISR rate 0.83 for both correct and incorrect claims). v2 was planned after seeing v1; it is reported *in addition to* v1, never instead of it. Three changes address suspected causes of the null result:

1. **Margin effect.** The effect of ablating the cited features is measured on the margin (log-odds of the claimed country against the 5 strongest competing countries), not on raw log-probability, to avoid saturation when the model is confident.
2. **Confidence control.** The added value of ISR is tested after adjusting for model confidence and template agreement.
3. **Held-out design.** v2 uses 800 *new* facts (v1 facts excluded), tail-weighted (100 head, 300 mid, 400 tail). 20% (dev) are used only to choose layer/k; the other 80% (test) are used for every reported test.

**Hypotheses (test split, chosen layer/k only).**
- H1: AUROC of margin-ISR support for correctness > 0.5 (bootstrap 95% CI excludes 0.5).
- H2 (primary): a likelihood-ratio test of logistic regression `correct ~ log(conf) + agreement + support` vs `correct ~ log(conf) + agreement` gives p < 0.05, with a positive support coefficient.
- H3: AUROC of support within confidence terciles is > 0.5 in at least two of three bands.
- H4: ISR rate is higher for correct than for incorrect claims (Fisher exact, two-sided p < 0.05); EPA rates reported alongside.

**Selection rule.** Among the configurations run, the (layer, k) with the highest dev AUROC of support is used for the test analysis. All configurations are also tabulated descriptively on the test split (not used for selection).

**Reporting rules.** No other settings are tried after the test split is analysed. If H2 fails, the conclusion is that, in this setting, feature-level support adds no information about correctness beyond confidence. That is a legitimate finding and will be stated as such. Secondary exploratory analyses are labelled exploratory.

**Known limits.** One model (Gemma 2 2B), one relation (located in), one SAE width/sparsity, ISR measures faithfulness to the model's mechanism rather than truth, and number of incorrect test claims (~150–250 expected) limits power.
