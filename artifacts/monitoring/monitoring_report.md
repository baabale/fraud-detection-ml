# Fraud Detection Model Monitoring Report

Generated on: 2025-05-27 13:00:55

## Model Performance

### Reference Data Performance

| Metric | Value |
| ------ | ----- |
| accuracy | 0.7000 |
| precision | 0.0000 |
| recall | 0.0000 |
| f1_score | 0.0000 |

**Note:** No positive predictions - model may need retraining

### Production Data Performance

| Metric | Value |
| ------ | ----- |
| accuracy | 0.8560 |
| precision | 0.0000 |
| recall | 0.0000 |
| f1_score | 0.0000 |

**Note:** No positive predictions - model may need retraining

### Performance Difference (Production - Reference)

| Metric | Difference |
| ------ | ---------- |
| accuracy | 0.1560 |
| precision | 0.0000 |
| recall | 0.0000 |
| f1_score | 0.0000 |

## Data Drift Analysis

Drift threshold (p-value): 0.05

Overall drift percentage: 91.67%

### Drifted Features

| Feature | KS Statistic | P-value |
| ------- | ------------ | ------- |
| amount | 0.3060 | 0.000000 |
| time_since_last_transaction | 0.9065 | 0.000000 |
| spending_deviation_score | 0.5200 | 0.000000 |
| velocity_score | 1.0000 | 0.000000 |
| geo_anomaly_score | 0.0840 | 0.017742 |
| amount_log | 0.3060 | 0.000000 |
| day_of_week | 0.2000 | 0.000000 |
| month | 0.6100 | 0.000000 |
| year | 1.0000 | 0.000000 |
| fraud_label | 0.1560 | 0.000000 |
| velocity_score_norm | 0.8360 | 0.000000 |

## Recommendations

- **Low data drift detected**: No immediate action required.
- **Model prediction issue**: The model is not predicting any positive cases. Consider adjusting the classification threshold or retraining with balanced data.
- **Unexpected performance improvement**: Model accuracy has increased by more than 10%. Verify that there is no data leakage or sampling bias in the production data.

---
*This report was generated automatically by the model monitoring system.*
