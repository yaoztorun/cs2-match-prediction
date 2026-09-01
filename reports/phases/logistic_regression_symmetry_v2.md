# Logistic Regression V2 - Side-Symmetry Diagnostic

Same diagnostic as RF and XGBoost: for every validation matchup A-vs-B, the raw mirrored B-vs-A form is built with the same `mirror_raw_rows`, transformed with the SAME fitted V2 preprocessing artifact, and scored with the same model.

Logistic Regression's linear form combined with mirrored training makes near-exact antisymmetry theoretically plausible, but this is **measured rather than assumed** - residual deviation can still arise from preprocessing (median imputation of a mirrored NaN pair shares one value) and floating-point effects.

- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n=1419 validation rows
- mean: 4.186e-17
- median: 0.000e+00
- 95th percentile: 1.110e-16
- max: 2.220e-16

**No external symmetry correction is applied.**
