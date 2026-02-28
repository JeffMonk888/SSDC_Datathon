# Personal Finance ML Pattern Report

Generated: 2026-02-28T15:52:51.892129+00:00

## Dataset
- Rows: 16241
- Columns: 19
- Modeling rows after filtering: 16241
- Target: `PWNETWPG`

## Supervised Model (Ridge Regression)
- Best alpha (5-fold CV): 25.0
- Train R2: 0.7037
- Test R2: 0.6939
- Test MAE: 605004.64
- Test RMSE: 1008756.62

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
- `metrics.json`
- `analysis_summary.md`