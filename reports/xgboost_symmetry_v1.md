# XGBoost V1 - Side-Symmetry Diagnostic

For every validation matchup A-vs-B, the raw mirrored B-vs-A form was built with the same `mirror_raw_rows` used for training augmentation, transformed with the SAME fitted XGBoost preprocessing artifact, and scored with the same model. For a perfectly side-consistent model, `P(A beats B) + P(B beats A) == 1`.

**No prior claim is made about the expected value of this error.** A gradient-boosted tree ensemble is *not* mathematically constrained to satisfy antisymmetry, even though its raw mirrored directional inputs are exact negatives, its symmetric/context inputs are unchanged, its mirrored training set is exactly balanced, and the model itself is deterministic - determinism does not imply antisymmetry. This is treated exactly like the Random Forest diagnostic: measured and reported, never assumed and never corrected in V1.

- symmetry_error = abs(P(A beats B) - (1 - P(B beats A))), n=1419 validation rows
- mean: 0.034423
- median: 0.026287
- 95th percentile: 0.098018
- max: 0.214942

**Observation**: a non-trivial asymmetry is present. A later phase could compare the raw probability against an explicit symmetrized probability `0.5 * [P(A beats B) + (1 - P(B beats A))]` - not implemented here.

**No probability symmetrization is applied in V1.**
