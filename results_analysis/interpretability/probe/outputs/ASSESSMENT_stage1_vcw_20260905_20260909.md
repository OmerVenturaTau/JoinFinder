# Probe assessment: `stage1_vcw_20260905`

This assessment uses a manuscript-disjoint 70/30 outer split (seed 42), with
five-fold tuning confined to the training manuscripts. The 11 configured
manuscript exclusions are applied before splitting. Reported confidence
intervals are non-parametric manuscript-level bootstrap intervals.

## Balance audit

The data are stratified, but the held-out sets intentionally retain their
natural imbalance:

- Oriental status: 1,209 manuscripts total. Train has 787 non-Oriental and 59
  Oriental manuscripts; test has 337 non-Oriental and 26 Oriental manuscripts.
  Oriental prevalence in the test set is only 7.2%, so raw accuracy is not an
  adequate success metric.
- Dating: the seven evaluable buckets contain between 16 and 325 manuscripts.
  Train/test counts by bucket are 11/5 (1200), 55/24 (1250), 75/32 (1300),
  81/35 (1350), 116/49 (1400), 227/98 (1450), and 70/30 (1500).
- Three dating buckets below the 10-manuscript threshold (1050, 1100, 1150;
  nine manuscripts total) are now excluded from fitting and evaluation. The
  earlier implementation kept them as train-only classes.

The original dating page/class weight product did not balance classes: effective
class shares ranged from 3.4% to 15.2%. It has been replaced with a joint weight
that gives each target class equal total weight, each manuscript equal weight
inside its class, and divides that manuscript weight equally over all its pages.
The Oriental logistic probe uses the same exact weighting rule. The final test
sets are not resampled or reweighted.

## Oriental logistic probe

| Space | Balanced accuracy (95% CI) | AUROC (95% CI) | Sensitivity | Specificity | PR-AUC |
|---|---:|---:|---:|---:|---:|
| tile | .612 (.532-.704) | .804 (.741-.862) | .269 | .955 | .233 |
| glyph | .590 (.512-.678) | .814 (.747-.872) | .231 | .950 | .249 |
| word | .726 (.622-.819) | .770 (.651-.876) | .654 | .798 | .392 |
| shared | **.754 (.658-.851)** | **.908 (.862-.950)** | .538 | .970 | **.497** |

The shared representation is a successful held-out linear discriminator. Its
PR-AUC of .497 is substantially above the 7.2% positive-prevalence baseline.
However, the default 0.5 threshold finds only 14 of 26 Oriental manuscripts and
misses 12. The word space is more sensitive (17/26) but produces many more false
positives. Therefore this probe is successful as evidence of linearly decodable
Oriental status and as a ranker, but the threshold is not ready for high-recall
classification without explicit threshold tuning.

Today's rerun reproduced the tile, glyph, and word prediction files byte for
byte. The shared fit is computationally expensive on CPU; the complete shared
result above is the existing run from 2026-09-08 with the same code, data, split,
and seed.

## Dating probes

The existing `run_dating_probe.py` is a small nonlinear MLP, not a linear
regression. After correcting its balance, its best exact-bucket test accuracy is
.333 (shared), below the majority-class baseline of .359. Its best macro-F1 is
.268 (word), and its best top-3 accuracy is .729 (word) versus .667 from always
choosing the three most frequent training buckets. The paired bootstrap interval
for that top-3 improvement includes zero (-.004 to .128). This is weak/mixed
evidence, not a successful exact-date classifier.

The added ridge probe directly tests linear date information after mean-pooling
pages within each manuscript:

| Space | MAE years | MAE change vs 1400 median baseline (95% CI) | R2 | Spearman rho |
|---|---:|---:|---:|---:|
| tile | 65.8 | +1.9 (-5.2 to +9.1) | -.134 | .365 |
| glyph | 65.8 | +1.8 (-3.4 to +7.0) | -.027 | .262 |
| word | 61.9 | -2.1 (-8.6 to +4.5) | .041 | .438 |
| shared | **55.5** | **-8.4 (-14.1 to -2.5)** | **.212** | **.500** |

Only the shared vector shows a statistically clear MAE improvement over the
63.9-year median baseline. Its nearest-bucket accuracy is .293 versus .179 for
that baseline. This is evidence of modest linear dating information, but not a
strong dating system: average error remains about 55 years and only 51.3% of
predictions fall within 50 years.

For the shared ridge model, the standardized prediction equation has intercept
1381.21 years. The largest coefficients are shared dimensions 233 (-3.061 years
per SD), 302 (+3.041), 642 (-2.673), 154 (-2.633), and 97 (+2.580). Positive
values move a prediction later and negative values move it earlier, conditional
on all other dimensions. Dimensions 0-511 are the tile block, 512-1023 the glyph
block, and 1024-1535 the word block for this checkpoint. Squared L2 coefficient
mass is 47.2% tile, 45.0% glyph, and 7.8% word. These latent-coordinate
coefficients are not directly interpretable as paleographic attributes and
should not be presented as independent causal effects.

## Interpretation and next checks

These probes establish decodability, not causality. Oriental status may still be
partly decoded through collection, imaging, background, or source-dataset cues;
dating may share the same confounds. Before making a stronger representation
claim, repeat the complete analysis over several manuscript split seeds, add a
label-permutation negative control, and report results stratified by library or
collection where metadata permit.
