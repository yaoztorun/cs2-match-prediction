# XGBoost V2 - Side-Symmetry Diagnostic

Same diagnostic as XGBoost V1 and Random Forest: for every validation matchup A-vs-B, the raw mirrored B-vs-A form is built with the same `mirror_raw_rows`, transformed with the SAME fitted V2 preprocessing artifact, and scored with the same model.

**No assumption is made that tuning improves symmetry.** A gradient-boosted tree ensemble is not mathematically constrained to satisfy `P(A beats B) = 1 - P(B beats A)`; determinism does not imply antisymmetry. Measured, reported, and left uncorrected.

- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n=1419 validation rows

| statistic | XGB V1 | XGB V2 | change |
|---|---|---|---|
| mean | 0.034423 | 0.009317 | -0.025105 |
| median | 0.026287 | 0.007694 | -0.018593 |
| 95th percentile | 0.098018 | 0.023101 | -0.074917 |
| max | 0.214942 | 0.058808 | -0.156134 |

**Observation**: V2's mean symmetry error is lower than V1's. Reported as observed - no symmetrization applied in either version.
