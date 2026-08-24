# Random Forest V1 - Side-Symmetry Diagnostic

For every validation matchup A-vs-B, the raw mirrored B-vs-A form was built with the same `mirror_raw_rows` used for training augmentation, transformed with the SAME fitted Random Forest preprocessing artifact, and scored with the same model. For a perfectly side-consistent model, `P(A beats B) + P(B beats A) == 1`. Random Forest is **not** mathematically guaranteed to satisfy this exactly even when trained on mirrored data (unlike Logistic Regression's linear structure, a tree ensemble's split boundaries need not be antisymmetric). This is diagnostic only - **no probability correction is applied in V1**.

- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n=1419 validation rows
- mean: 0.0311
- median: 0.0267
- 95th percentile: 0.0733
- max: 0.1533

**Observation**: asymmetry is small in this run, but is still not exactly zero by construction (unlike Logistic Regression) - reported for the record, not corrected.
