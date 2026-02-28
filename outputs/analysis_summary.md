# Personal Finance ML Pattern Report

Generated: 2026-02-28T21:11:47.912153+00:00

## Dataset
- Rows: 16241
- Columns: 19
- Modeling rows after filtering: 16241
- Target: `PWNETWPG`

## Phase 1: Supervised Model (Ridge Regression on PWNETWPG)
- Best alpha (5-fold CV): 25.0
- Train R2: 0.7037
- Test R2: 0.6939
- Test MAE: 605004.64
- Test RMSE: 1008756.62

## Phase 2: Financial Resilience Score (FRS)
- Design: robust-scaled weighted score using liquidity, savings gap, equity ratio, income, DTI, and credit card debt.
- Baseline tier thresholds: q20=38.72, q50=48.05, q80=63.84
- Baseline tier shares: At Risk=20.00%, Coping=30.00%, Stable=30.00%, Thriving=20.00%
- Corr(FRS, Net Worth): 0.4157
- Corr(FRS, Anomaly Distance): 0.3169
- FRS-only Net Worth R2: 0.1728

Resilience tier median profile:
```text
                   count  share  PEFATINC   PWNETWPG  TOTAL_DEBT  LIQUID_ASSETS  DEBT_TO_INCOME  LIQUIDITY_TO_INCOME  HOME_EQUITY_TO_VALUE  SAVINGS_GAP  PWDSTCRD
FRS_TIER_BASELINE                                                                                                                                                
At Risk             3249    0.2   88750.0   477200.0    212500.0         5000.0            2.42                 0.06                  0.49          0.0    7000.0
Coping              4872    0.3   63375.0   412050.0      7250.0         5250.0            0.13                 0.09                  0.86       4750.0       0.0
Stable              4872    0.3   98150.0  1248375.0         0.0        59000.0            0.00                 0.67                  1.00      58500.0       0.0
Thriving            3248    0.2   94662.5  2311000.0         0.0       245000.0            0.00                 2.82                  1.00     245000.0       0.0
```

Stress scenario impacts:
```text
           scenario  mean_frs_post  mean_frs_drop  median_frs_drop  pct_below_baseline_fragility_cutoff  most_impacted_cluster  most_impacted_cluster_mean_drop  share_at_risk_post  share_coping_post  share_stable_post  share_thriving_post
    rate_hike_200bp        50.8937         0.1370           0.0078                               0.2049                      0                           0.4724              0.2049             0.2979             0.2979               0.1994
 income_shock_20pct        50.9560         0.0748           0.3512                               0.2211                      0                           1.6097              0.2211             0.2970             0.2695               0.2124
housing_shock_15pct        50.7460         0.2848           0.0000                               0.2123                      0                           0.8555              0.2123             0.2928             0.2957               0.1992
```

Top positive factors (higher value tends to increase predicted net worth):
```text
       feature                                    readable_feature      beta_std
   HOME_EQUITY                 Home Equity (Home Value - Mortgage) 551232.616641
      PEFATINC                                    After-Tax Income 429156.063965
      PWAPRVAL                                          Home Value 342093.368057
     PFMTYPG=9                                         Family Type 340630.403410
   SAVINGS_GAP                      Savings Minus Credit Card Debt 334266.059881
 LIQUID_ASSETS                     Liquid Assets (Deposits + TFSA) 313813.208608
DEBT_TO_INCOME                                Debt-to-Income Ratio 100556.697978
    PAGEMIEG=5                                           Age Group  76264.807644
    TOTAL_DEBT Total Debt (Mortgage + Student + Credit Card + LOC)  71311.976087
    PAGEMIEG=6                                           Age Group  61452.132584
```

Top negative factors (higher value tends to decrease predicted net worth):
```text
            feature          readable_feature       beta_std
           PWDPRMOR             Mortgage Debt -240805.127205
           PWASTDEP             Bank Deposits -211439.309283
         PAGEMIEG=2                 Age Group  -95413.381593
          PFMTYPG=4               Family Type  -87230.016449
          PFMTYPG=3               Family Type  -67687.162116
          PFMTYPG=2               Family Type  -65031.463259
          PPVRES=11                  Province  -64091.525429
LIQUIDITY_TO_INCOME Liquidity-to-Income Ratio  -58395.423956
          PPVRES=10                  Province  -54591.591407
          PPVRES=59                  Province  -54486.433220
```

## Household Segmentation (KMeans)
- Best K by silhouette: 2
- Silhouette scores: {'2': 0.3938842249945553, '3': 0.35570322769134655, '4': 0.28798321414896544, '5': 0.3257298921810394, '6': 0.3292240601100006}

Cluster median profile:
```text
         count  share  PEFATINC  PWAPRVAL  PWASTDEP  PWDPRMOR  PWDSTCRD   PWNETWPG  TOTAL_DEBT  LIQUID_ASSETS  DEBT_TO_INCOME
CLUSTER                                                                                                                      
0         2982   0.18  127050.0  775000.0    8000.0  310000.0     287.5  1076852.5    354000.0        18500.0            3.06
1        13259   0.82   77025.0  310000.0   11500.0       0.0       0.0   861500.0         0.0        33750.0            0.00
```

## Anomaly Detection
- Method: Mahalanobis distance on key balance-sheet and income features.
- Exported top 30 anomalous rows to `anomalies.csv`.

## Files Produced
- `engineered_dataset_with_cluster.csv`
- `factor_loadings.csv`
- `cluster_profiles.csv`
- `anomalies.csv`
- `resilience_scores.csv`
- `resilience_tier_summary.csv`
- `resilience_cluster_summary.csv`
- `resilience_scenario_summary.csv`
- `resilience_transition_matrix.csv`
- `resilience_bootstrap_stability.json`
- `metrics.json`
- `analysis_summary.md`