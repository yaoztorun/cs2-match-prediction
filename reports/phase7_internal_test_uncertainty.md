# Phase 7 - Sealed Internal TEST Uncertainty (Cluster Bootstrap)

Predefined contract, frozen before TEST was opened: match_id-cluster bootstrap, N_BOOTSTRAP=2000, RANDOM_STATE=42, 95% percentile intervals. Reads ONLY `data/evaluation/map_test_predictions_v1.parquet`.

## Final model: 95% cluster-bootstrap intervals

| metric | point estimate | 95% CI lower | 95% CI upper | valid replicates |
|---|---|---|---|---|
| accuracy | 0.6132 | 0.5884 | 0.6400 | 2000/2000 |
| roc_auc | 0.6489 | 0.6197 | 0.6789 | 2000/2000 |
| log_loss | 0.6521 | 0.6398 | 0.6640 | 2000/2000 |
| brier | 0.2301 | 0.2241 | 0.2358 | 2000/2000 |

AUC: 2000/2000 replicates were valid (single-class replicates skipped and counted separately, per the predefined contract).

## Paired bootstrap deltas vs. baselines (identical per-replicate draws)

Positive favors the final XGB for Accuracy/ROC-AUC; negative favors it for Log Loss/Brier.

| comparison | metric | observed Δ | 95% CI lower | 95% CI upper | favors final XGB |
|---|---|---|---|---|---|
| final_xgb_v3_minus_overall_elo | accuracy | +0.0000 | -0.0162 | +0.0158 | False |
| final_xgb_v3_minus_overall_elo | roc_auc | +0.0012 | -0.0117 | +0.0133 | True |
| final_xgb_v3_minus_overall_elo | log_loss | -0.0028 | -0.0125 | +0.0071 | True |
| final_xgb_v3_minus_overall_elo | brier | -0.0013 | -0.0053 | +0.0028 | True |
| final_xgb_v3_minus_map_elo | accuracy | +0.0519 | +0.0243 | +0.0792 | True |
| final_xgb_v3_minus_map_elo | roc_auc | +0.0626 | +0.0351 | +0.0884 | True |
| final_xgb_v3_minus_map_elo | log_loss | -0.0265 | -0.0369 | -0.0154 | True |
| final_xgb_v3_minus_map_elo | brier | -0.0127 | -0.0176 | -0.0075 | True |

This is inferential/descriptive evaluation of the already-frozen model - it is not a model-selection mechanism, and no configuration changes follow from it.
