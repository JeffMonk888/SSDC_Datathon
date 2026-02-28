#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.stats import chi2

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

NAMESPACES = {"m": MAIN_NS, "r": REL_NS, "pr": PKG_REL_NS}


def col_letters_to_index(col_letters: str) -> int:
    idx = 0
    for char in col_letters:
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def read_shared_strings(zip_file: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zip_file.namelist():
        return []

    root = ET.fromstring(zip_file.read("xl/sharedStrings.xml"))
    out: List[str] = []
    for shared_item in root.findall("m:si", NAMESPACES):
        text_nodes = shared_item.findall(".//m:t", NAMESPACES)
        out.append("".join(node.text or "" for node in text_nodes))
    return out


def workbook_sheet_paths(zip_file: zipfile.ZipFile) -> Dict[str, str]:
    workbook = ET.fromstring(zip_file.read("xl/workbook.xml"))
    workbook_rels = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))

    rel_id_to_target: Dict[str, str] = {}
    for rel in workbook_rels.findall("pr:Relationship", NAMESPACES):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if not rel_id:
            continue
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = f"xl/{target}"
        rel_id_to_target[rel_id] = target

    sheet_paths: Dict[str, str] = {}
    for sheet in workbook.findall("m:sheets/m:sheet", NAMESPACES):
        name = sheet.attrib.get("name")
        rel_id = sheet.attrib.get(f"{{{REL_NS}}}id")
        if not name or not rel_id:
            continue
        target = rel_id_to_target.get(rel_id)
        if target:
            sheet_paths[name] = target
    return sheet_paths


def decode_cell_value(cell: ET.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("m:v", NAMESPACES)

    if cell_type == "s":
        if value_node is None or value_node.text is None:
            return ""
        return shared_strings[int(value_node.text)]

    if cell_type == "inlineStr":
        text_node = cell.find("m:is/m:t", NAMESPACES)
        return text_node.text if text_node is not None and text_node.text else ""

    if cell_type == "b":
        if value_node is None:
            return ""
        return "1" if value_node.text == "1" else "0"

    return value_node.text if value_node is not None and value_node.text is not None else ""


def parse_sheet(zip_file: zipfile.ZipFile, sheet_path: str, shared_strings: Sequence[str]) -> pd.DataFrame:
    root = ET.fromstring(zip_file.read(sheet_path))
    sheet_data = root.find("m:sheetData", NAMESPACES)
    if sheet_data is None:
        return pd.DataFrame()

    sparse_rows: List[Dict[int, str]] = []
    max_col = -1

    for row in sheet_data.findall("m:row", NAMESPACES):
        sparse_row: Dict[int, str] = {}
        for cell in row.findall("m:c", NAMESPACES):
            cell_ref = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)", cell_ref)
            if not match:
                continue
            col_idx = col_letters_to_index(match.group(1))
            max_col = max(max_col, col_idx)
            sparse_row[col_idx] = decode_cell_value(cell, shared_strings)
        sparse_rows.append(sparse_row)

    if max_col < 0 or not sparse_rows:
        return pd.DataFrame()

    matrix: List[List[str]] = []
    for sparse_row in sparse_rows:
        matrix.append([sparse_row.get(i, "") for i in range(max_col + 1)])

    header = matrix[0]
    data = matrix[1:]
    columns: List[str] = []
    for i, name in enumerate(header):
        clean_name = str(name).strip()
        if not clean_name:
            clean_name = f"col_{i}"
        columns.append(clean_name)

    return pd.DataFrame(data, columns=columns)


def load_xlsx_data(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(xlsx_path) as zip_file:
        shared_strings = read_shared_strings(zip_file)
        sheet_paths = workbook_sheet_paths(zip_file)

        dictionary_sheet = "dictionary"
        data_sheet = "datathon_finance"

        if dictionary_sheet not in sheet_paths or data_sheet not in sheet_paths:
            all_sheet_names = list(sheet_paths.keys())
            if len(all_sheet_names) < 2:
                raise ValueError("Workbook needs at least two sheets (dictionary + data).")
            dictionary_sheet = all_sheet_names[0]
            data_sheet = all_sheet_names[1]

        dict_df = parse_sheet(zip_file, sheet_paths[dictionary_sheet], shared_strings)
        data_df = parse_sheet(zip_file, sheet_paths[data_sheet], shared_strings)

    return dict_df, data_df


def build_variable_metadata(dict_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Variable Name", "Readable Name", "Description", "Type", "Values/Ranges", "Notes"]
    for col in required_cols:
        if col not in dict_df.columns:
            dict_df[col] = ""

    meta = dict_df[required_cols].copy()
    meta["Variable Name"] = meta["Variable Name"].astype(str).str.strip()
    meta = meta[meta["Variable Name"] != ""]
    meta = meta[meta["Variable Name"].str.upper() != "NAN"]
    meta = meta.drop_duplicates(subset=["Variable Name"])
    return meta.reset_index(drop=True)


def sum_existing(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return df[existing].sum(axis=1, min_count=1)


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["TOTAL_DEBT"] = sum_existing(out, ["PWDPRMOR", "PWDSLOAN", "PWDSTCRD", "PWDSTLOC"])
    out["LIQUID_ASSETS"] = sum_existing(out, ["PWASTDEP", "PWATFS"])
    out["HOME_EQUITY"] = safe_series(out, "PWAPRVAL") - safe_series(out, "PWDPRMOR")

    income_abs = safe_series(out, "PEFATINC").abs().replace(0, np.nan)
    out["DEBT_TO_INCOME"] = out["TOTAL_DEBT"] / income_abs
    out["LIQUIDITY_TO_INCOME"] = out["LIQUID_ASSETS"] / income_abs

    home_value = safe_series(out, "PWAPRVAL").replace(0, np.nan)
    out["HOME_EQUITY_TO_VALUE"] = out["HOME_EQUITY"] / home_value
    out["SAVINGS_GAP"] = out["LIQUID_ASSETS"] - safe_series(out, "PWDSTCRD")
    return out


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce")
    if clean.notna().sum() == 0:
        return clean
    low = clean.quantile(lower_q)
    high = clean.quantile(upper_q)
    return clean.clip(lower=low, upper=high)


@dataclass
class RidgeModel:
    beta_std: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    alpha: float


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> RidgeModel:
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std == 0] = 1.0
    X_std = (X - x_mean) / x_std

    y_mean = float(y.mean())
    y_centered = y - y_mean

    n_features = X_std.shape[1]
    reg = alpha * np.eye(n_features)
    beta_std = np.linalg.solve(X_std.T @ X_std + reg, X_std.T @ y_centered)

    return RidgeModel(beta_std=beta_std, x_mean=x_mean, x_std=x_std, y_mean=y_mean, alpha=alpha)


def predict_ridge(model: RidgeModel, X: np.ndarray) -> np.ndarray:
    X_std = (X - model.x_mean) / model.x_std
    return model.y_mean + X_std @ model.beta_std


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def cv_select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Sequence[float],
    folds: int = 5,
    seed: int = 42,
) -> Tuple[float, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    fold_indices = np.array_split(indices, folds)

    alpha_scores: Dict[str, float] = {}
    for alpha in alphas:
        fold_scores: List[float] = []
        for i in range(folds):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(folds) if j != i])

            model = fit_ridge(X[train_idx], y[train_idx], alpha=alpha)
            preds = predict_ridge(model, X[val_idx])
            fold_scores.append(r2_score(y[val_idx], preds))

        alpha_scores[str(alpha)] = float(np.nanmean(fold_scores))

    best_alpha = max(alphas, key=lambda a: alpha_scores[str(a)] if not math.isnan(alpha_scores[str(a)]) else -np.inf)
    return float(best_alpha), alpha_scores


def kmeans_with_restarts(X: np.ndarray, k: int, seed: int, restarts: int = 8) -> Tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    best_centroids: np.ndarray | None = None
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")

    for _ in range(restarts):
        run_seed = int(rng.integers(0, 1_000_000_000))
        try:
            centroids, labels = kmeans2(X, k, minit="points", iter=50, seed=run_seed)
        except TypeError:
            centroids, labels = kmeans2(X, k, minit="points", iter=50)

        inertia = float(np.sum((X - centroids[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    if best_centroids is None or best_labels is None:
        raise RuntimeError("KMeans failed to produce any clustering result.")

    return best_centroids, best_labels, best_inertia


def silhouette_sample(X: np.ndarray, labels: np.ndarray, seed: int, sample_size: int = 1200) -> float:
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[idx]
        labels_sample = labels[idx]
    else:
        X_sample = X
        labels_sample = labels

    diffs = X_sample[:, None, :] - X_sample[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))

    scores: List[float] = []
    for i in range(len(X_sample)):
        own_label = labels_sample[i]

        same_mask = labels_sample == own_label
        same_mask[i] = False

        if np.any(same_mask):
            a_i = float(dists[i, same_mask].mean())
        else:
            a_i = 0.0

        b_i = float("inf")
        for label in unique_labels:
            if label == own_label:
                continue
            other_mask = labels_sample == label
            if np.any(other_mask):
                b_i = min(b_i, float(dists[i, other_mask].mean()))

        denom = max(a_i, b_i)
        if not np.isfinite(denom) or denom == 0:
            scores.append(0.0)
        else:
            scores.append((b_i - a_i) / denom)

    return float(np.mean(scores)) if scores else float("nan")


def choose_cluster_count(X: np.ndarray, seed: int, k_min: int = 2, k_max: int = 6) -> Tuple[int, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for k in range(k_min, k_max + 1):
        _, labels, _ = kmeans_with_restarts(X, k=k, seed=seed + k)
        score = silhouette_sample(X, labels, seed=seed + (10 * k))
        scores[str(k)] = score

    best_k = max(range(k_min, k_max + 1), key=lambda k: scores[str(k)] if not math.isnan(scores[str(k)]) else -np.inf)
    return int(best_k), scores


def compute_mahalanobis_distances(X: np.ndarray) -> np.ndarray:
    center = X.mean(axis=0)
    centered = X - center
    cov = np.cov(centered, rowvar=False)
    ridge = (np.trace(cov) / cov.shape[0]) * 1e-6 if cov.shape[0] > 0 else 1e-6
    cov = cov + np.eye(cov.shape[0]) * ridge
    cov_inv = np.linalg.pinv(cov)
    return np.einsum("ij,jk,ik->i", centered, cov_inv, centered)


def summarize_feature_name(feature: str, readable_map: Dict[str, str], engineered_map: Dict[str, str]) -> str:
    if feature in engineered_map:
        return engineered_map[feature]
    if "=" in feature:
        base = feature.split("=", 1)[0]
        return readable_map.get(base, base)
    return readable_map.get(feature, feature)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_resilience_component_specs(df: pd.DataFrame, component_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    specs: Dict[str, Dict[str, float]] = {}
    for col in component_cols:
        raw = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        if raw.notna().sum() == 0:
            specs[col] = {"lower": np.nan, "upper": np.nan, "median": 0.0, "iqr": 1.0, "fill": 0.0}
            continue

        lower = float(raw.quantile(0.01))
        upper = float(raw.quantile(0.99))
        clipped = raw.clip(lower=lower, upper=upper)

        median = float(clipped.median()) if clipped.notna().sum() > 0 else 0.0
        q1 = float(clipped.quantile(0.25)) if clipped.notna().sum() > 0 else 0.0
        q3 = float(clipped.quantile(0.75)) if clipped.notna().sum() > 0 else 0.0
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            iqr = 1.0
        fill = median if np.isfinite(median) else 0.0
        specs[col] = {"lower": lower, "upper": upper, "median": median, "iqr": iqr, "fill": fill}

    return specs


def transform_component_with_spec(series: pd.Series, spec: Dict[str, float]) -> Tuple[pd.Series, pd.Series]:
    s = pd.to_numeric(series, errors="coerce")
    lower = spec.get("lower", np.nan)
    upper = spec.get("upper", np.nan)
    if np.isfinite(lower):
        s = s.clip(lower=lower)
    if np.isfinite(upper):
        s = s.clip(upper=upper)

    fill_value = spec.get("fill", 0.0)
    s = s.fillna(fill_value)

    median = spec.get("median", 0.0)
    iqr = spec.get("iqr", 1.0)
    if not np.isfinite(iqr) or iqr == 0:
        iqr = 1.0
    z = (s - median) / iqr
    return s.astype(float), z.astype(float)


def compute_resilience_score(
    df: pd.DataFrame,
    specs: Dict[str, Dict[str, float]],
    positive_weights: Dict[str, float],
    negative_weights: Dict[str, float],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    linear = np.zeros(len(df), dtype=float)

    # Keep transformed component values and robust z-scores for auditability.
    for feature, weight in positive_weights.items():
        raw = df[feature] if feature in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        component, z = transform_component_with_spec(raw, specs[feature])
        out[f"FRS_COMP_{feature}"] = component
        out[f"FRS_Z_{feature}"] = z
        linear += weight * z.to_numpy()

    for feature, weight in negative_weights.items():
        raw = df[feature] if feature in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
        component, z = transform_component_with_spec(raw, specs[feature])
        out[f"FRS_COMP_{feature}"] = component
        out[f"FRS_Z_{feature}"] = z
        linear -= weight * z.to_numpy()

    out["FRS_LINEAR"] = linear
    out["FRS_SCORE"] = 100.0 * sigmoid(linear)
    return out


def derive_tier_thresholds(scores: pd.Series) -> Dict[str, float]:
    q20 = float(scores.quantile(0.20))
    q50 = float(scores.quantile(0.50))
    q80 = float(scores.quantile(0.80))
    return {"q20": q20, "q50": q50, "q80": q80}


def assign_resilience_tiers(scores: pd.Series, thresholds: Dict[str, float]) -> pd.Series:
    bins = [-np.inf, thresholds["q20"], thresholds["q50"], thresholds["q80"], np.inf]
    labels = ["At Risk", "Coping", "Stable", "Thriving"]
    tiers = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
    return tiers.astype("string").fillna("Unknown")


def apply_resilience_scenario(
    base_df: pd.DataFrame,
    scenario_kind: str,
    *,
    rate_delta: float = 0.02,
    rate_kappa: float = 5.0,
    income_shock: float = 0.20,
    housing_shock: float = 0.15,
) -> pd.DataFrame:
    scenario_df = base_df.copy()

    if scenario_kind == "rate_hike":
        dti_multiplier = 1.0 + (rate_delta * rate_kappa)
        scenario_df["DEBT_TO_INCOME"] = safe_series(scenario_df, "DEBT_TO_INCOME") * dti_multiplier
        return scenario_df

    if scenario_kind == "income_shock":
        income_prime = safe_series(scenario_df, "PEFATINC") * (1.0 - income_shock)
        scenario_df["PEFATINC"] = income_prime

        income_abs = income_prime.abs().replace(0, np.nan)
        scenario_df["DEBT_TO_INCOME"] = safe_series(scenario_df, "TOTAL_DEBT") / income_abs
        scenario_df["LIQUIDITY_TO_INCOME"] = safe_series(scenario_df, "LIQUID_ASSETS") / income_abs
        return scenario_df

    if scenario_kind == "housing_shock":
        home_value_prime = safe_series(scenario_df, "PWAPRVAL") * (1.0 - housing_shock)
        scenario_df["PWAPRVAL"] = home_value_prime

        mortgage = safe_series(scenario_df, "PWDPRMOR")
        home_equity_prime = home_value_prime - mortgage
        scenario_df["HOME_EQUITY"] = home_equity_prime
        scenario_df["HOME_EQUITY_TO_VALUE"] = home_equity_prime / home_value_prime.replace(0, np.nan)
        return scenario_df

    raise ValueError(f"Unknown scenario_kind={scenario_kind}")


def build_tier_transition(
    baseline_tier: pd.Series,
    scenario_tier: pd.Series,
    scenario_name: str,
) -> pd.DataFrame:
    tier_order = ["At Risk", "Coping", "Stable", "Thriving"]
    matrix = pd.crosstab(
        baseline_tier,
        scenario_tier,
        dropna=False,
    ).reindex(index=tier_order, columns=tier_order, fill_value=0)
    matrix.index.name = "baseline_tier"
    matrix.columns.name = "scenario_tier"

    transition_long = matrix.stack().rename("count").reset_index()
    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    transition_long["share_within_baseline_tier"] = transition_long.apply(
        lambda r: float(r["count"] / row_sums.loc[r["baseline_tier"]]) if pd.notna(row_sums.loc[r["baseline_tier"]]) else 0.0,
        axis=1,
    )
    transition_long["scenario"] = scenario_name
    return transition_long


def simple_linear_regression_r2(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    mask = x_num.notna() & y_num.notna()
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "intercept": float("nan"), "slope": float("nan"), "r2": float("nan")}

    x_arr = x_num.loc[mask].to_numpy(dtype=float)
    y_arr = y_num.loc[mask].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(x_arr)), x_arr])
    beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
    y_hat = X @ beta
    return {
        "n": int(len(x_arr)),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": r2_score(y_arr, y_hat),
    }


def bootstrap_tier_stability(scores: pd.Series, iterations: int = 200, seed: int = 42) -> Dict[str, Any]:
    arr = pd.to_numeric(scores, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return {
            "iterations": iterations,
            "threshold_mean": {"q20": float("nan"), "q50": float("nan"), "q80": float("nan")},
            "threshold_std": {"q20": float("nan"), "q50": float("nan"), "q80": float("nan")},
            "tier_share_mean": {"At Risk": float("nan"), "Coping": float("nan"), "Stable": float("nan"), "Thriving": float("nan")},
            "tier_share_std": {"At Risk": float("nan"), "Coping": float("nan"), "Stable": float("nan"), "Thriving": float("nan")},
        }

    rng = np.random.default_rng(seed)
    tier_order = ["At Risk", "Coping", "Stable", "Thriving"]

    threshold_records: List[Dict[str, float]] = []
    share_records: List[Dict[str, float]] = []

    for _ in range(iterations):
        sample = arr[rng.integers(0, len(arr), size=len(arr))]
        q20, q50, q80 = np.quantile(sample, [0.20, 0.50, 0.80])
        threshold_records.append({"q20": float(q20), "q50": float(q50), "q80": float(q80)})

        tiers = pd.cut(
            sample,
            bins=[-np.inf, q20, q50, q80, np.inf],
            labels=tier_order,
            include_lowest=True,
        )
        shares = pd.Series(tiers).value_counts(normalize=True).reindex(tier_order, fill_value=0.0)
        share_records.append({k: float(v) for k, v in shares.items()})

    threshold_df = pd.DataFrame(threshold_records)
    share_df = pd.DataFrame(share_records)

    return {
        "iterations": iterations,
        "threshold_mean": {k: float(v) for k, v in threshold_df.mean().items()},
        "threshold_std": {k: float(v) for k, v in threshold_df.std(ddof=1).items()},
        "tier_share_mean": {k: float(v) for k, v in share_df.mean().items()},
        "tier_share_std": {k: float(v) for k, v in share_df.std(ddof=1).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline ML pattern mining for personal finance dataset.")
    parser.add_argument("--xlsx", default="personal_finance_dataset.xlsx", help="Path to source XLSX file.")
    parser.add_argument("--outdir", default="outputs", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bootstrap_iters", type=int, default=200, help="Bootstrap iterations for resilience stability.")
    parser.add_argument("--rate_delta", type=float, default=0.02, help="Rate shock in absolute terms (e.g., 0.02 for +200 bps).")
    parser.add_argument("--rate_kappa", type=float, default=5.0, help="Pass-through sensitivity from rates to DTI.")
    parser.add_argument("--income_shock", type=float, default=0.20, help="Income shock fraction (e.g., 0.20 for -20%).")
    parser.add_argument("--housing_shock", type=float, default=0.15, help="Housing price shock fraction (e.g., 0.15 for -15%).")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    dict_df_raw, data_df_raw = load_xlsx_data(xlsx_path)
    meta_df = build_variable_metadata(dict_df_raw)

    readable_map = {
        row["Variable Name"]: row["Readable Name"]
        for _, row in meta_df.iterrows()
        if str(row["Variable Name"]).strip() != ""
    }
    type_map = {
        row["Variable Name"]: str(row["Type"]).strip().lower()
        for _, row in meta_df.iterrows()
        if str(row["Variable Name"]).strip() != ""
    }

    data_df = data_df_raw.copy()
    for col in data_df.columns:
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

    engineered_df = build_engineered_features(data_df)
    target_col = "PWNETWPG"
    if target_col not in engineered_df.columns:
        raise ValueError(f"Target column {target_col} not found.")

    categorical_cols = sorted(
        [
            c
            for c in data_df.columns
            if c != target_col and ("categorical" in type_map.get(c, "") or "binary" in type_map.get(c, ""))
        ]
    )
    continuous_cols = sorted(
        [c for c in data_df.columns if c != target_col and "continuous" in type_map.get(c, "")]
    )

    engineered_continuous_cols = [
        "TOTAL_DEBT",
        "LIQUID_ASSETS",
        "HOME_EQUITY",
        "DEBT_TO_INCOME",
        "LIQUIDITY_TO_INCOME",
        "HOME_EQUITY_TO_VALUE",
        "SAVINGS_GAP",
    ]

    continuous_model_cols = [c for c in (continuous_cols + engineered_continuous_cols) if c in engineered_df.columns]

    X_cont = engineered_df[continuous_model_cols].copy()
    for col in X_cont.columns:
        X_cont[col] = winsorize_series(X_cont[col])
        X_cont[col] = X_cont[col].fillna(X_cont[col].median())

    X_cat_raw = engineered_df[categorical_cols].copy() if categorical_cols else pd.DataFrame(index=engineered_df.index)
    X_cat = pd.DataFrame(index=engineered_df.index)
    if not X_cat_raw.empty:
        for col in X_cat_raw.columns:
            cleaned = pd.to_numeric(X_cat_raw[col], errors="coerce").round()
            as_int = cleaned.astype("Int64").astype(str)
            as_int = as_int.replace("<NA>", "MISSING")
            X_cat_raw[col] = as_int
        X_cat = pd.get_dummies(X_cat_raw, prefix=categorical_cols, prefix_sep="=", dtype=float)

    X_df = pd.concat([X_cont, X_cat], axis=1)
    y_series = winsorize_series(engineered_df[target_col]).astype(float)

    valid_rows = y_series.notna() & np.isfinite(X_df.to_numpy()).all(axis=1)
    X_model = X_df.loc[valid_rows]
    y_model = y_series.loc[valid_rows]

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(X_model))
    rng.shuffle(all_indices)
    train_cutoff = int(0.8 * len(all_indices))
    train_idx = all_indices[:train_cutoff]
    test_idx = all_indices[train_cutoff:]

    X_array = X_model.to_numpy(dtype=float)
    y_array = y_model.to_numpy(dtype=float)

    X_train, y_train = X_array[train_idx], y_array[train_idx]
    X_test, y_test = X_array[test_idx], y_array[test_idx]

    alpha_grid = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    best_alpha, cv_scores = cv_select_alpha(X_train, y_train, alpha_grid, folds=5, seed=args.seed)

    model = fit_ridge(X_train, y_train, alpha=best_alpha)
    train_preds = predict_ridge(model, X_train)
    test_preds = predict_ridge(model, X_test)

    feature_names = X_model.columns.to_list()
    coeff_df = pd.DataFrame(
        {
            "feature": feature_names,
            "beta_std": model.beta_std,
        }
    )
    coeff_df["abs_beta_std"] = coeff_df["beta_std"].abs()

    engineered_readable_map = {
        "TOTAL_DEBT": "Total Debt (Mortgage + Student + Credit Card + LOC)",
        "LIQUID_ASSETS": "Liquid Assets (Deposits + TFSA)",
        "HOME_EQUITY": "Home Equity (Home Value - Mortgage)",
        "DEBT_TO_INCOME": "Debt-to-Income Ratio",
        "LIQUIDITY_TO_INCOME": "Liquidity-to-Income Ratio",
        "HOME_EQUITY_TO_VALUE": "Home Equity as Share of Home Value",
        "SAVINGS_GAP": "Savings Minus Credit Card Debt",
    }
    coeff_df["readable_feature"] = coeff_df["feature"].apply(
        lambda f: summarize_feature_name(f, readable_map, engineered_readable_map)
    )
    coeff_df = coeff_df.sort_values("abs_beta_std", ascending=False).reset_index(drop=True)

    cluster_features = [
        "PEFATINC",
        "PWAPRVAL",
        "PWASTDEP",
        "PWATFS",
        "PWDPRMOR",
        "PWDSLOAN",
        "PWDSTCRD",
        "PWDSTLOC",
        "TOTAL_DEBT",
        "LIQUID_ASSETS",
        "DEBT_TO_INCOME",
        "LIQUIDITY_TO_INCOME",
    ]
    cluster_features = [c for c in cluster_features if c in engineered_df.columns]

    X_cluster_df = engineered_df[cluster_features].copy()
    for col in X_cluster_df.columns:
        X_cluster_df[col] = winsorize_series(X_cluster_df[col])
        X_cluster_df[col] = X_cluster_df[col].fillna(X_cluster_df[col].median())

    X_cluster = X_cluster_df.to_numpy(dtype=float)
    cluster_mean = X_cluster.mean(axis=0)
    cluster_std = X_cluster.std(axis=0)
    cluster_std[cluster_std == 0] = 1.0
    X_cluster_std = (X_cluster - cluster_mean) / cluster_std

    best_k, silhouette_scores = choose_cluster_count(X_cluster_std, seed=args.seed, k_min=2, k_max=6)
    _, cluster_labels, cluster_inertia = kmeans_with_restarts(
        X_cluster_std,
        k=best_k,
        seed=args.seed + 999,
        restarts=10,
    )

    clustered_df = engineered_df.copy()
    clustered_df["CLUSTER"] = cluster_labels.astype(int)

    profile_cols = [
        "PEFATINC",
        "PWAPRVAL",
        "PWASTDEP",
        "PWDPRMOR",
        "PWDSTCRD",
        "PWNETWPG",
        "TOTAL_DEBT",
        "LIQUID_ASSETS",
        "DEBT_TO_INCOME",
    ]
    profile_cols = [c for c in profile_cols if c in clustered_df.columns]

    cluster_profiles = clustered_df.groupby("CLUSTER")[profile_cols].median()
    cluster_counts = clustered_df["CLUSTER"].value_counts().sort_index()
    cluster_profiles.insert(0, "count", cluster_counts)
    cluster_profiles.insert(1, "share", (cluster_counts / cluster_counts.sum()).round(4))

    anomaly_features = [
        "PEFATINC",
        "PWAPRVAL",
        "TOTAL_DEBT",
        "LIQUID_ASSETS",
        "DEBT_TO_INCOME",
        "PWNETWPG",
    ]
    anomaly_features = [c for c in anomaly_features if c in engineered_df.columns]
    X_anom_df = engineered_df[anomaly_features].copy()
    for col in X_anom_df.columns:
        X_anom_df[col] = winsorize_series(X_anom_df[col])
        X_anom_df[col] = X_anom_df[col].fillna(X_anom_df[col].median())

    md2 = compute_mahalanobis_distances(X_anom_df.to_numpy(dtype=float))
    p_values = 1.0 - chi2.cdf(md2, df=len(anomaly_features))

    anomaly_table = engineered_df.copy()
    anomaly_table["mahalanobis_sq"] = md2
    anomaly_table["chi2_p_value"] = p_values
    anomaly_table["source_row"] = anomaly_table.index + 2
    anomaly_table = anomaly_table.sort_values("mahalanobis_sq", ascending=False)
    top_anomalies = anomaly_table.head(30)

    # -----------------------------
    # Resilience Score (FRS) Engine
    # -----------------------------
    resilience_positive_weights = {
        "LIQUIDITY_TO_INCOME": 0.30,
        "SAVINGS_GAP": 0.20,
        "HOME_EQUITY_TO_VALUE": 0.20,
        "PEFATINC": 0.15,
    }
    resilience_negative_weights = {
        "DEBT_TO_INCOME": 0.10,
        "PWDSTCRD": 0.05,
    }

    resilience_component_cols = sorted(
        set(list(resilience_positive_weights.keys()) + list(resilience_negative_weights.keys()))
    )
    resilience_specs = fit_resilience_component_specs(clustered_df, resilience_component_cols)
    resilience_baseline_df = compute_resilience_score(
        clustered_df,
        specs=resilience_specs,
        positive_weights=resilience_positive_weights,
        negative_weights=resilience_negative_weights,
    )
    frs_baseline = resilience_baseline_df["FRS_SCORE"]

    tier_thresholds = derive_tier_thresholds(frs_baseline)
    frs_tier_baseline = assign_resilience_tiers(frs_baseline, tier_thresholds)

    clustered_df["FRS_BASELINE"] = frs_baseline
    clustered_df["FRS_TIER_BASELINE"] = frs_tier_baseline

    tier_order = ["At Risk", "Coping", "Stable", "Thriving"]
    tier_counts = frs_tier_baseline.value_counts().reindex(tier_order, fill_value=0)
    tier_shares = (tier_counts / max(1, tier_counts.sum())).astype(float)

    resilience_tier_profile_cols = [
        "PEFATINC",
        "PWNETWPG",
        "TOTAL_DEBT",
        "LIQUID_ASSETS",
        "DEBT_TO_INCOME",
        "LIQUIDITY_TO_INCOME",
        "HOME_EQUITY_TO_VALUE",
        "SAVINGS_GAP",
        "PWDSTCRD",
    ]
    resilience_tier_profile_cols = [c for c in resilience_tier_profile_cols if c in clustered_df.columns]
    resilience_tier_summary = clustered_df.groupby("FRS_TIER_BASELINE")[resilience_tier_profile_cols].median()
    resilience_tier_summary = resilience_tier_summary.reindex(tier_order)
    resilience_tier_summary.insert(0, "count", tier_counts)
    resilience_tier_summary.insert(1, "share", tier_shares.round(4))

    cluster_resilience_base = (
        clustered_df.groupby("CLUSTER")["FRS_BASELINE"]
        .agg(["count", "mean", "median"])
        .rename(columns={"mean": "frs_mean", "median": "frs_median"})
    )
    cluster_resilience_base["share_dataset"] = (cluster_resilience_base["count"] / cluster_resilience_base["count"].sum()).round(4)
    tier_share_by_cluster = pd.crosstab(
        clustered_df["CLUSTER"],
        clustered_df["FRS_TIER_BASELINE"],
        normalize="index",
    ).reindex(columns=tier_order, fill_value=0.0)
    cluster_resilience_summary = cluster_resilience_base.join(
        tier_share_by_cluster.add_prefix("tier_share_"),
        how="left",
    )

    frs_networth_corr = float(pd.Series(frs_baseline).corr(clustered_df[target_col], method="pearson"))
    frs_anomaly_corr = float(pd.Series(frs_baseline).corr(pd.Series(md2, index=clustered_df.index), method="pearson"))
    frs_regression_diag = simple_linear_regression_r2(frs_baseline, clustered_df[target_col])
    bootstrap_stats = bootstrap_tier_stability(frs_baseline, iterations=args.bootstrap_iters, seed=args.seed + 73)

    scenario_definitions = [
        {"name": "rate_hike_200bp", "kind": "rate_hike"},
        {"name": f"income_shock_{int(round(args.income_shock * 100))}pct", "kind": "income_shock"},
        {"name": f"housing_shock_{int(round(args.housing_shock * 100))}pct", "kind": "housing_shock"},
    ]

    resilience_scenario_rows: List[Dict[str, Any]] = []
    resilience_transition_rows: List[pd.DataFrame] = []
    resilience_scenario_wide = pd.DataFrame(index=clustered_df.index)
    resilience_scenario_wide["source_row"] = clustered_df.index + 2
    resilience_scenario_wide["FRS_BASELINE"] = frs_baseline
    resilience_scenario_wide["FRS_TIER_BASELINE"] = frs_tier_baseline

    for scenario in scenario_definitions:
        scenario_name = scenario["name"]
        scenario_kind = scenario["kind"]

        scenario_df = apply_resilience_scenario(
            clustered_df,
            scenario_kind=scenario_kind,
            rate_delta=args.rate_delta,
            rate_kappa=args.rate_kappa,
            income_shock=args.income_shock,
            housing_shock=args.housing_shock,
        )
        scenario_score_df = compute_resilience_score(
            scenario_df,
            specs=resilience_specs,
            positive_weights=resilience_positive_weights,
            negative_weights=resilience_negative_weights,
        )

        scenario_score = scenario_score_df["FRS_SCORE"]
        scenario_tier = assign_resilience_tiers(scenario_score, tier_thresholds)
        score_drop = frs_baseline - scenario_score

        scenario_score_col = f"FRS_{scenario_name}"
        scenario_delta_col = f"FRS_DELTA_{scenario_name}"
        scenario_tier_col = f"FRS_TIER_{scenario_name}"

        resilience_scenario_wide[scenario_score_col] = scenario_score
        resilience_scenario_wide[scenario_delta_col] = score_drop
        resilience_scenario_wide[scenario_tier_col] = scenario_tier

        resilience_transition_rows.append(
            build_tier_transition(frs_tier_baseline, scenario_tier, scenario_name=scenario_name)
        )

        impacted_cluster = pd.DataFrame({"cluster": clustered_df["CLUSTER"], "score_drop": score_drop}).groupby("cluster")[
            "score_drop"
        ].mean()
        if len(impacted_cluster) > 0:
            worst_cluster = int(impacted_cluster.idxmax())
            worst_cluster_drop = float(impacted_cluster.max())
        else:
            worst_cluster = -1
            worst_cluster_drop = float("nan")

        post_tier_shares = scenario_tier.value_counts(normalize=True).reindex(tier_order, fill_value=0.0)
        resilience_scenario_rows.append(
            {
                "scenario": scenario_name,
                "mean_frs_post": float(scenario_score.mean()),
                "mean_frs_drop": float(score_drop.mean()),
                "median_frs_drop": float(score_drop.median()),
                "pct_below_baseline_fragility_cutoff": float((scenario_score <= tier_thresholds["q20"]).mean()),
                "most_impacted_cluster": worst_cluster,
                "most_impacted_cluster_mean_drop": worst_cluster_drop,
                "share_at_risk_post": float(post_tier_shares.get("At Risk", 0.0)),
                "share_coping_post": float(post_tier_shares.get("Coping", 0.0)),
                "share_stable_post": float(post_tier_shares.get("Stable", 0.0)),
                "share_thriving_post": float(post_tier_shares.get("Thriving", 0.0)),
            }
        )

    resilience_scenario_summary = pd.DataFrame(resilience_scenario_rows)
    resilience_transition_matrix = (
        pd.concat(resilience_transition_rows, ignore_index=True)
        if resilience_transition_rows
        else pd.DataFrame(columns=["baseline_tier", "scenario_tier", "count", "share_within_baseline_tier", "scenario"])
    )

    resilience_scores = pd.DataFrame(index=clustered_df.index)
    resilience_scores["source_row"] = clustered_df.index + 2
    resilience_scores["CLUSTER"] = clustered_df["CLUSTER"]
    resilience_scores["FRS_BASELINE"] = frs_baseline
    resilience_scores["FRS_LINEAR_BASELINE"] = resilience_baseline_df["FRS_LINEAR"]
    resilience_scores["FRS_TIER_BASELINE"] = frs_tier_baseline
    for component in resilience_component_cols:
        resilience_scores[f"FRS_COMP_{component}"] = resilience_baseline_df[f"FRS_COMP_{component}"]
        resilience_scores[f"FRS_Z_{component}"] = resilience_baseline_df[f"FRS_Z_{component}"]

    key_context_cols = [
        "PEFATINC",
        "PWNETWPG",
        "TOTAL_DEBT",
        "LIQUID_ASSETS",
        "DEBT_TO_INCOME",
        "LIQUIDITY_TO_INCOME",
        "HOME_EQUITY_TO_VALUE",
        "SAVINGS_GAP",
    ]
    for col in key_context_cols:
        if col in clustered_df.columns:
            resilience_scores[col] = clustered_df[col]

    resilience_scores = resilience_scores.merge(
        resilience_scenario_wide,
        how="left",
        left_index=True,
        right_index=True,
        suffixes=("", "_dup"),
    )
    resilience_scores = resilience_scores.loc[:, ~resilience_scores.columns.str.endswith("_dup")]

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "rows_total": int(len(data_df)),
            "columns_total": int(data_df.shape[1]),
            "rows_modeling": int(len(X_model)),
            "target": target_col,
        },
        "model": {
            "type": "ridge_regression_closed_form",
            "alpha_grid": alpha_grid,
            "best_alpha": best_alpha,
            "cv_r2_by_alpha": cv_scores,
            "train_r2": r2_score(y_train, train_preds),
            "test_r2": r2_score(y_test, test_preds),
            "train_mae": mae(y_train, train_preds),
            "test_mae": mae(y_test, test_preds),
            "train_rmse": rmse(y_train, train_preds),
            "test_rmse": rmse(y_test, test_preds),
        },
        "clustering": {
            "algorithm": "kmeans",
            "candidate_k": [2, 3, 4, 5, 6],
            "silhouette_by_k": silhouette_scores,
            "best_k": best_k,
            "inertia": cluster_inertia,
        },
        "anomaly_detection": {
            "method": "mahalanobis_distance",
            "features": anomaly_features,
            "top_anomalies_exported": 30,
        },
        "resilience": {
            "score_name": "Financial Resilience Score (FRS)",
            "weights": {
                "positive": resilience_positive_weights,
                "negative": resilience_negative_weights,
            },
            "tier_thresholds": tier_thresholds,
            "tier_counts": {k: int(v) for k, v in tier_counts.items()},
            "tier_shares": {k: float(v) for k, v in tier_shares.items()},
            "validation": {
                "corr_frs_vs_net_worth": frs_networth_corr,
                "corr_frs_vs_mahalanobis_sq": frs_anomaly_corr,
                "frs_only_regression": frs_regression_diag,
            },
            "bootstrap_stability": bootstrap_stats,
            "scenario_summary": resilience_scenario_rows,
        },
    }

    top_pos = coeff_df.sort_values("beta_std", ascending=False).head(10)[
        ["feature", "readable_feature", "beta_std"]
    ]
    top_neg = coeff_df.sort_values("beta_std", ascending=True).head(10)[
        ["feature", "readable_feature", "beta_std"]
    ]

    summary_lines = [
        "# Personal Finance ML Pattern Report",
        "",
        f"Generated: {metrics['generated_utc']}",
        "",
        "## Dataset",
        f"- Rows: {metrics['dataset']['rows_total']}",
        f"- Columns: {metrics['dataset']['columns_total']}",
        f"- Modeling rows after filtering: {metrics['dataset']['rows_modeling']}",
        f"- Target: `{target_col}`",
        "",
        "## Phase 1: Supervised Model (Ridge Regression on PWNETWPG)",
        f"- Best alpha (5-fold CV): {best_alpha}",
        f"- Train R2: {metrics['model']['train_r2']:.4f}",
        f"- Test R2: {metrics['model']['test_r2']:.4f}",
        f"- Test MAE: {metrics['model']['test_mae']:.2f}",
        f"- Test RMSE: {metrics['model']['test_rmse']:.2f}",
        "",
        "## Phase 2: Financial Resilience Score (FRS)",
        "- Design: robust-scaled weighted score using liquidity, savings gap, equity ratio, income, DTI, and credit card debt.",
        f"- Baseline tier thresholds: q20={tier_thresholds['q20']:.2f}, q50={tier_thresholds['q50']:.2f}, q80={tier_thresholds['q80']:.2f}",
        f"- Baseline tier shares: At Risk={tier_shares['At Risk']:.2%}, Coping={tier_shares['Coping']:.2%}, Stable={tier_shares['Stable']:.2%}, Thriving={tier_shares['Thriving']:.2%}",
        f"- Corr(FRS, Net Worth): {frs_networth_corr:.4f}",
        f"- Corr(FRS, Anomaly Distance): {frs_anomaly_corr:.4f}",
        f"- FRS-only Net Worth R2: {frs_regression_diag['r2']:.4f}",
        "",
        "Resilience tier median profile:",
        "```text",
        resilience_tier_summary.round(2).to_string(),
        "```",
        "",
        "Stress scenario impacts:",
        "```text",
        resilience_scenario_summary.round(4).to_string(index=False),
        "```",
        "",
        "Top positive factors (higher value tends to increase predicted net worth):",
        "```text",
        top_pos.to_string(index=False),
        "```",
        "",
        "Top negative factors (higher value tends to decrease predicted net worth):",
        "```text",
        top_neg.to_string(index=False),
        "```",
        "",
        "## Household Segmentation (KMeans)",
        f"- Best K by silhouette: {best_k}",
        f"- Silhouette scores: {silhouette_scores}",
        "",
        "Cluster median profile:",
        "```text",
        cluster_profiles.round(2).to_string(),
        "```",
        "",
        "## Anomaly Detection",
        "- Method: Mahalanobis distance on key balance-sheet and income features.",
        "- Exported top 30 anomalous rows to `anomalies.csv`.",
        "",
        "## Files Produced",
        "- `engineered_dataset_with_cluster.csv`",
        "- `factor_loadings.csv`",
        "- `cluster_profiles.csv`",
        "- `anomalies.csv`",
        "- `resilience_scores.csv`",
        "- `resilience_tier_summary.csv`",
        "- `resilience_cluster_summary.csv`",
        "- `resilience_scenario_summary.csv`",
        "- `resilience_transition_matrix.csv`",
        "- `resilience_bootstrap_stability.json`",
        "- `metrics.json`",
        "- `analysis_summary.md`",
    ]

    (outdir / "engineered_dataset_with_cluster.csv").write_text(
        clustered_df.to_csv(index=False),
        encoding="utf-8",
    )
    coeff_df.to_csv(outdir / "factor_loadings.csv", index=False)
    cluster_profiles.round(4).to_csv(outdir / "cluster_profiles.csv")
    top_anomalies.to_csv(outdir / "anomalies.csv", index=False)
    resilience_scores.to_csv(outdir / "resilience_scores.csv", index=False)
    resilience_tier_summary.round(4).to_csv(outdir / "resilience_tier_summary.csv")
    cluster_resilience_summary.round(4).to_csv(outdir / "resilience_cluster_summary.csv")
    resilience_scenario_summary.round(6).to_csv(outdir / "resilience_scenario_summary.csv", index=False)
    resilience_transition_matrix.to_csv(outdir / "resilience_transition_matrix.csv", index=False)
    (outdir / "resilience_bootstrap_stability.json").write_text(
        json.dumps(bootstrap_stats, indent=2),
        encoding="utf-8",
    )
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (outdir / "analysis_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    meta_df.to_csv(outdir / "dictionary_cleaned.csv", index=False)

    print(f"Finished. Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
