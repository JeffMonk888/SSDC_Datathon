# SSDC Datathon - Personal Finance ML Pipeline

This repo now contains an offline-capable ML workflow to mine patterns from:

- `Personal Finance Case.pdf`
- `personal_finance_dataset.xlsx`

The pipeline does:

- XLSX ingestion without `openpyxl` (parses workbook XML directly)
- Feature engineering (debt, liquidity, home equity, ratios)
- Supervised net-worth modeling (`PWNETWPG`) via ridge regression with CV
- Unsupervised household segmentation via KMeans
- Anomaly detection via Mahalanobis distance
- Export of tables and a markdown summary report

## Quickstart

1. (Optional) create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

or run:

```bash
./scripts/setup_env.sh
```

3. Run analysis:

```bash
python3 analysis/run_finance_ml.py --xlsx personal_finance_dataset.xlsx --outdir outputs --seed 42
```

## Outputs

The script writes these files to `outputs/`:

- `metrics.json`
- `analysis_summary.md`
- `factor_loadings.csv`
- `cluster_profiles.csv`
- `anomalies.csv`
- `engineered_dataset_with_cluster.csv`
- `dictionary_cleaned.csv`

## Notes

- In this environment, internet package download is blocked, so `pip install` may fail here.
- The analysis code itself is written to run fully offline once required packages are available.
