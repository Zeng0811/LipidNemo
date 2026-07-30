#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCA-aware mechanistic interpretability analysis for LipidNemo models.

This script is designed for models trained on PCA-compressed embedding features
plus ratio features. It keeps the interpretation anchored in the model's actual
training feature space instead of treating the original 2560 embedding
dimensions as the primary explanation space.

Analyses performed:
1. Robust Mordred descriptor calculation for the ionizable lipid SMILES column.
2. Permutation importance in the model's PCA/model feature space.
3. Correlation mapping from important PCA/model features to Mordred descriptors.
4. Mordred-surrogate SHAP analysis for model target-class scores.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score

from analyze_lipidnemo_interpretability import (
    AutoGluonPredictorWrapper,
    COMPONENT_NAMES,
    MORDRED_CACHE_METADATA_FILE,
    MORDRED_CLEANED_CACHE_FILE,
    compute_and_clean_mordred_descriptors,
    compute_tree_shap_values,
    configure_logging,
    ensure_optional_dependencies,
    get_target_scores,
    load_cached_mordred_descriptors,
    load_and_align_inputs,
    load_autogluon_predictor,
    make_full_feature_names,
    parse_label_mapping,
    resolve_target_values,
    save_mordred_cache_metadata,
    safe_path_token,
    summarize_shap_importance,
)


LOGGER = logging.getLogger("LipidNemoInterpretabilityPCA")
RATIO_COLS = len(COMPONENT_NAMES)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run PCA-aware mechanistic interpretability analysis for LipidNemo."
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=repo_dir / "LNP-447-ft-final.npy",
        help="Path to the raw (N, 2565) embedding .npy file.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=repo_dir / "LNP-447.CSV",
        help="Path to the metadata CSV file containing labels and SMILES.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=repo_dir / "TabPFNModels" / "m-20260707_114852_406642",
        help=(
            "Path to a trained PCA-based model. Supported inputs are an "
            "AutoGluon predictor directory with utils/data/X.pkl or a saved "
            "joblib/pkl package containing scaler/pca/model."
        ),
    )
    parser.add_argument(
        "--require_model",
        action="store_true",
        help="Fail instead of training a fallback proxy if --model_path is unusable.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=script_dir / "PCA_embedding-analysis",
        help="Directory where all CSV, PNG and metadata outputs will be saved.",
    )
    parser.add_argument(
        "--smiles_col",
        default="Ionizable lipid(S)",
        help="CSV column containing ionizable lipid SMILES.",
    )
    parser.add_argument(
        "--label_col",
        default="Organ",
        help="CSV column containing organ-targeting labels.",
    )
    parser.add_argument(
        "--label_mapping",
        default="Liver:0,Spleen:1,Lung:2,None:3",
        help="Comma-separated label mapping, for example 'Liver:0,Spleen:1,Lung:2,None:3'.",
    )
    parser.add_argument(
        "--missing_label_value",
        default="",
        help="Optional class label used to replace missing/blank labels in --label_col.",
    )
    parser.add_argument(
        "--target_class",
        default="2",
        help="Class label for Mordred SHAP target score, or 'all' to analyze every observed class.",
    )
    parser.add_argument(
        "--mordred_missing_threshold",
        type=float,
        default=0.10,
        help="Drop Mordred descriptor columns with missing fraction above this threshold.",
    )
    parser.add_argument(
        "--variance_threshold",
        type=float,
        default=1e-12,
        help="VarianceThreshold cutoff for cleaned Mordred descriptors.",
    )
    parser.add_argument(
        "--force_recompute_mordred",
        action="store_true",
        help="Ignore cached Mordred descriptors and recompute them from SMILES.",
    )
    parser.add_argument(
        "--mordred_cache_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing previously computed Mordred cache files "
            "(for example a prior results directory with mordred_cleaned_descriptors.csv "
            "and mordred_cache_metadata.json)."
        ),
    )
    parser.add_argument(
        "--train_sample_ids_csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing a 'sample_id' column for the model training subset. "
            "When provided, raw rows/Mordred descriptors are subset and reordered to match "
            "the saved model feature matrix."
        ),
    )
    parser.add_argument(
        "--permutation_scoring",
        default="accuracy",
        help="Scoring metric passed to sklearn.inspection.permutation_importance.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of repeats for permutation importance.",
    )
    parser.add_argument(
        "--proxy_n_estimators",
        type=int,
        default=600,
        help="Number of trees for fallback classifier and surrogate regressors.",
    )
    parser.add_argument(
        "--top_model_features",
        type=int,
        default=20,
        help="Number of top PCA/model features to highlight and map to Mordred descriptors.",
    )
    parser.add_argument(
        "--top_mordred_descriptors",
        type=int,
        default=20,
        help="Number of top Mordred descriptors to show in the PCA-feature correlation heatmap.",
    )
    parser.add_argument(
        "--top_shap_features",
        type=int,
        default=25,
        help="Number of top Mordred SHAP features to draw in the bar plot.",
    )
    parser.add_argument(
        "--correlation_labeled_only",
        action="store_true",
        help="Use only labeled samples when correlating PCA/model features with Mordred descriptors.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for Mordred, permutation importance and surrogate trees.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def default_model_feature_names(
    feature_count: int,
    ratio_cols: int = RATIO_COLS,
    feature_mode: str = "pca",
) -> List[str]:
    if feature_count < ratio_cols:
        return [f"ModelFeature_{idx:04d}" for idx in range(feature_count)]

    core_count = feature_count - ratio_cols
    if feature_mode == "pca":
        core_names = [f"PCA_PC_{idx + 1:04d}" for idx in range(core_count)]
    elif feature_mode == "raw":
        core_names = [f"Embedding_feature_{idx:04d}" for idx in range(core_count)]
    else:
        core_names = [f"ModelFeature_{idx:04d}" for idx in range(core_count)]
    ratio_names = [f"Ratio__{name}" for name in COMPONENT_NAMES[:ratio_cols]]
    return core_names + ratio_names


def standardize_feature_columns(
    feature_df: pd.DataFrame,
    ratio_cols: int = RATIO_COLS,
    feature_mode: str = "pca",
) -> pd.DataFrame:
    columns = list(feature_df.columns)
    if not columns:
        return feature_df

    normalized: List[Optional[int]] = []
    for col in columns:
        if isinstance(col, (int, np.integer)):
            normalized.append(int(col))
            continue
        text = str(col).strip()
        if text.isdigit():
            normalized.append(int(text))
            continue
        normalized.append(None)

    is_plain_range = normalized == list(range(len(columns)))
    if is_plain_range:
        feature_df = feature_df.copy()
        feature_df.columns = default_model_feature_names(
            feature_count=len(columns),
            ratio_cols=ratio_cols,
            feature_mode=feature_mode,
        )
        return feature_df

    feature_df = feature_df.copy()
    feature_df.columns = [str(col) for col in columns]
    return feature_df


def prepare_features_with_saved_package(
    x_raw: np.ndarray,
    save_package: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    scaler = save_package.get("scaler")
    pca = save_package.get("pca")
    pca_meta = save_package.get("pca_meta", {}) or {}
    ratio_cols = int(pca_meta.get("ratio_cols", RATIO_COLS))
    emb_dim = pca_meta.get("emb_dim", save_package.get("emb_dim"))
    feature_mode = save_package.get("feature_mode", pca_meta.get("feature_mode", "pca"))

    if emb_dim is None:
        if scaler is not None and hasattr(scaler, "n_features_in_"):
            emb_dim = int(scaler.n_features_in_)
        elif scaler is not None and hasattr(scaler, "mean_"):
            emb_dim = int(len(scaler.mean_))
        else:
            emb_dim = int(x_raw.shape[1] - ratio_cols)
    else:
        emb_dim = int(emb_dim)

    expected_total_dim = emb_dim + ratio_cols
    if x_raw.shape[1] != expected_total_dim:
        raise ValueError(
            "Raw embedding dimension mismatch for the saved package: "
            f"input={x_raw.shape[1]}, expected={expected_total_dim} "
            f"(emb_dim={emb_dim}, ratio_cols={ratio_cols})."
        )

    x_emb = np.nan_to_num(
        x_raw[:, :emb_dim],
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    ).astype(np.float32)
    x_rat = np.nan_to_num(
        x_raw[:, emb_dim:],
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).astype(np.float32)

    if feature_mode == "raw":
        x_final = np.hstack([x_emb, x_rat]).astype(np.float32)
    else:
        if scaler is None or pca is None:
            raise KeyError("Saved PCA package must contain both 'scaler' and 'pca'.")
        x_emb_scaled = scaler.transform(x_emb)
        x_pca = pca.transform(x_emb_scaled)
        x_final = np.hstack([x_pca, x_rat]).astype(np.float32)

    feature_df = pd.DataFrame(
        x_final,
        columns=default_model_feature_names(
            feature_count=x_final.shape[1],
            ratio_cols=ratio_cols,
            feature_mode=feature_mode,
        ),
    )
    metadata = {
        "feature_mode": feature_mode,
        "emb_dim": emb_dim,
        "ratio_cols": ratio_cols,
        "output_dim": int(x_final.shape[1]),
        "saved_pca_meta": pca_meta,
    }
    return feature_df, metadata


def load_mordred_descriptors_with_optional_external_cache(
    smiles: pd.Series,
    output_dir: Path,
    missing_threshold: float,
    variance_threshold: float,
    n_jobs: int,
    force_recompute: bool,
    cache_dir: Optional[Path],
) -> pd.DataFrame:
    if not force_recompute and cache_dir is not None:
        cache_dir = cache_dir.resolve()
        output_dir_resolved = output_dir.resolve()
        if cache_dir != output_dir_resolved:
            LOGGER.info("Trying to reuse Mordred cache from external directory %s", cache_dir)
            cached_desc = load_cached_mordred_descriptors(
                smiles=smiles,
                output_dir=cache_dir,
                missing_threshold=missing_threshold,
                variance_threshold=variance_threshold,
            )
            if cached_desc is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                cached_desc.to_csv(output_dir / MORDRED_CLEANED_CACHE_FILE, index_label="sample_id")
                save_mordred_cache_metadata(
                    smiles=smiles,
                    output_dir=output_dir,
                    missing_threshold=missing_threshold,
                    variance_threshold=variance_threshold,
                    descriptor_columns=list(cached_desc.columns),
                )

                for filename in [
                    "mordred_raw_descriptors.csv",
                    "mordred_dropped_missing_columns.csv",
                    "mordred_dropped_low_variance_columns.csv",
                    "invalid_smiles.csv",
                ]:
                    src = cache_dir / filename
                    dst = output_dir / filename
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)

                LOGGER.info(
                    "Reused Mordred cache from %s and synchronized cleaned outputs into %s",
                    cache_dir,
                    output_dir,
                )
                return cached_desc

            LOGGER.info(
                "No reusable Mordred cache was found in %s. Falling back to the current output directory/cache build.",
                cache_dir,
            )

    return compute_and_clean_mordred_descriptors(
        smiles=smiles,
        output_dir=output_dir,
        missing_threshold=missing_threshold,
        variance_threshold=variance_threshold,
        n_jobs=n_jobs,
        force_recompute=force_recompute,
    )


def load_training_sample_ids(sample_ids_csv: Path) -> pd.Series:
    df_ids = pd.read_csv(sample_ids_csv)
    if "sample_id" not in df_ids.columns:
        raise KeyError(
            f"{sample_ids_csv} must contain a 'sample_id' column. "
            f"Available columns: {list(df_ids.columns)}"
        )
    sample_ids = pd.to_numeric(df_ids["sample_id"], errors="raise").astype(int)
    if sample_ids.duplicated().any():
        dup_values = sample_ids[sample_ids.duplicated()].unique().tolist()
        raise ValueError(
            f"{sample_ids_csv} contains duplicated sample_id values, which would break row alignment: {dup_values[:10]}"
        )
    return sample_ids.reset_index(drop=True)


def subset_aligned_rows_by_sample_ids(
    sample_ids: pd.Series,
    df_work: pd.DataFrame,
    smiles: pd.Series,
    labels: pd.Series,
    labeled_mask: pd.Series,
    mordred_desc: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    if sample_ids.empty:
        raise ValueError("The provided training sample_id list is empty.")

    max_id = int(sample_ids.max())
    min_id = int(sample_ids.min())
    if min_id < 0 or max_id >= len(df_work):
        raise ValueError(
            "Training sample_id values are out of range for the aligned raw dataset: "
            f"min={min_id}, max={max_id}, aligned_rows={len(df_work)}."
        )

    subset_df = df_work.iloc[sample_ids.to_numpy()].reset_index(drop=True)
    subset_smiles = smiles.iloc[sample_ids.to_numpy()].reset_index(drop=True)
    subset_labels = labels.iloc[sample_ids.to_numpy()].reset_index(drop=True)
    subset_labeled_mask = labeled_mask.iloc[sample_ids.to_numpy()].reset_index(drop=True)
    subset_mordred = mordred_desc.iloc[sample_ids.to_numpy()].reset_index(drop=True)
    return subset_df, subset_smiles, subset_labels, subset_labeled_mask, subset_mordred


def load_model_and_feature_space(
    model_path: Optional[Path],
    require_model: bool,
    x_raw: np.ndarray,
    labels: pd.Series,
    labeled_mask: pd.Series,
    random_state: int,
    n_estimators: int,
    n_jobs: int,
) -> Tuple[Any, pd.DataFrame, pd.Series, Dict[str, Any]]:
    raw_feature_names = make_full_feature_names()
    if model_path is not None and model_path.is_dir():
        if not (model_path / "predictor.pkl").exists():
            raise FileNotFoundError(f"predictor.pkl was not found in {model_path}")

        LOGGER.info("Loading AutoGluon predictor and saved training feature space from %s", model_path)
        model = load_autogluon_predictor(model_path, raw_feature_names)
        if model.training_X is None:
            raise ValueError(
                f"AutoGluon model directory {model_path} does not contain a readable utils/data/X.pkl."
            )
        feature_df = model.training_X.copy().reset_index(drop=True)
        feature_df = standardize_feature_columns(feature_df, ratio_cols=RATIO_COLS, feature_mode="pca")
        train_y = model.training_y
        if train_y is None:
            if require_model:
                raise ValueError(
                    f"AutoGluon model directory {model_path} does not contain a readable utils/data/y.pkl."
                )
            LOGGER.warning(
                "AutoGluon model directory has no readable y.pkl. Falling back to current CSV labels "
                "only if row counts match."
            )
            if len(feature_df) != len(labels):
                raise ValueError(
                    "Model feature matrix row count does not match current labels, and training y.pkl is missing."
                )
            train_y = labels.copy()
        else:
            train_y = pd.Series(train_y).reset_index(drop=True)

        metadata = {
            "model_source": "autogluon_predictor_directory",
            "feature_space_source": "utils/data/X.pkl",
            "feature_mode": "pca_or_saved_model_space",
            "raw_feature_count": int(x_raw.shape[1]),
            "model_feature_count": int(feature_df.shape[1]),
        }
        return model, feature_df, train_y, metadata

    if model_path is not None and model_path.is_file():
        LOGGER.info("Loading saved sklearn/joblib package from %s", model_path)
        loaded = joblib.load(model_path)

        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
            feature_df, package_meta = prepare_features_with_saved_package(x_raw, loaded)
            train_y = labels.copy().reset_index(drop=True)
            metadata = {
                "model_source": "saved_package_file",
                "feature_space_source": "reconstructed_from_saved_scaler_pca",
                "feature_mode": package_meta["feature_mode"],
                "raw_feature_count": int(x_raw.shape[1]),
                "model_feature_count": int(feature_df.shape[1]),
                "saved_pca_meta": package_meta["saved_pca_meta"],
            }
            return model, feature_df, train_y, metadata

        model = loaded
        feature_df = pd.DataFrame(x_raw, columns=raw_feature_names)
        train_y = labels.copy().reset_index(drop=True)
        metadata = {
            "model_source": "loaded_estimator_file",
            "feature_space_source": "raw_embedding_input",
            "feature_mode": "raw",
            "raw_feature_count": int(x_raw.shape[1]),
            "model_feature_count": int(feature_df.shape[1]),
        }
        return model, feature_df, train_y, metadata

    if require_model:
        raise ValueError("--require_model was set but a usable --model_path was not provided.")

    LOGGER.warning(
        "No usable PCA model was supplied. Training a fallback ExtraTrees classifier on the raw 2565 features."
    )
    x_labeled = x_raw[labeled_mask.to_numpy()]
    y_labeled = labels.loc[labeled_mask].to_numpy()
    model = ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced",
        max_features="sqrt",
    )
    model.fit(x_labeled, y_labeled)
    feature_df = pd.DataFrame(x_raw, columns=raw_feature_names)
    train_y = labels.copy().reset_index(drop=True)
    metadata = {
        "model_source": "fallback_extratrees_proxy",
        "feature_space_source": "raw_embedding_input",
        "feature_mode": "raw",
        "raw_feature_count": int(x_raw.shape[1]),
        "model_feature_count": int(feature_df.shape[1]),
    }
    return model, feature_df, train_y, metadata


def make_model_row_mask(
    feature_df: pd.DataFrame,
    labels_for_features: pd.Series,
    aligned_labels: pd.Series,
    aligned_labeled_mask: pd.Series,
    labeled_only: bool,
) -> np.ndarray:
    if not labeled_only:
        return np.ones(len(feature_df), dtype=bool)

    if len(feature_df) == len(aligned_labeled_mask):
        return aligned_labeled_mask.to_numpy()

    return pd.Series(labels_for_features).map(lambda value: not pd.isna(value)).to_numpy(dtype=bool)


def run_model_space_permutation_importance(
    model: Any,
    feature_df: pd.DataFrame,
    labels_for_features: pd.Series,
    output_dir: Path,
    scoring: str,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
    top_model_features: int,
) -> pd.DataFrame:
    y_series = pd.Series(labels_for_features).reset_index(drop=True)
    label_mask = y_series.map(lambda value: not pd.isna(value)).to_numpy(dtype=bool)
    x_perm = feature_df.loc[label_mask].reset_index(drop=True)
    y_perm = y_series.loc[label_mask].to_numpy()

    LOGGER.info(
        "Running permutation importance in model feature space: X=%s, y=%s",
        x_perm.shape,
        y_perm.shape,
    )
    permutation_n_jobs = n_jobs
    if isinstance(model, AutoGluonPredictorWrapper) and n_jobs != 1:
        LOGGER.warning(
            "AutoGluon/TabPFN prediction may allocate CUDA resources. "
            "Forcing permutation importance n_jobs=1 instead of %s.",
            n_jobs,
        )
        permutation_n_jobs = 1
    result = permutation_importance(
        model,
        x_perm,
        y_perm,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=permutation_n_jobs,
    )

    importance_df = pd.DataFrame(
        {
            "feature_index": np.arange(feature_df.shape[1]),
            "feature_name": list(feature_df.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(output_dir / "model_space_permutation_importance_all_features.csv", index=False)

    top_df = importance_df.head(top_model_features).reset_index(drop=True)
    top_df.to_csv(output_dir / "model_space_permutation_importance_top_features.csv", index=False)
    return top_df


def run_global_mordred_multiclass_importance(
    mordred_desc: pd.DataFrame,
    formulation_features: pd.DataFrame,
    labels: pd.Series,
    output_dir: Path,
    scoring: str,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
    n_estimators: int,
    top_features: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    label_series = pd.Series(labels).reset_index(drop=True)
    labeled_mask = label_series.map(lambda value: not pd.isna(value)).to_numpy(dtype=bool)
    x_mordred = mordred_desc.loc[labeled_mask].reset_index(drop=True)
    x_formulation = formulation_features.loc[labeled_mask].reset_index(drop=True)
    x_combined = pd.concat([x_mordred, x_formulation], axis=1)
    
    y_mordred = (
        label_series.loc[labeled_mask]
        .reset_index(drop=True)
        .map(lambda value: str(value).strip())
    )

    surrogate = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
        max_features="sqrt",
        min_samples_leaf=2,
    )
    surrogate.fit(x_combined, y_mordred)
    fitted = pd.Series(surrogate.predict(x_combined))
    acc = accuracy_score(y_mordred, fitted)
    balanced_acc = balanced_accuracy_score(y_mordred, fitted)

    LOGGER.info(
        "Running global Mordred multiclass permutation importance (controlled): X=%s, y=%s, accuracy=%.4f, balanced_accuracy=%.4f",
        x_combined.shape,
        y_mordred.shape,
        float(acc),
        float(balanced_acc),
    )
    result = permutation_importance(
        surrogate,
        x_combined,
        y_mordred,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance_df = pd.DataFrame(
        {
            "feature_index": np.arange(x_combined.shape[1]),
            "feature_name": list(x_combined.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    mordred_cols = set(x_mordred.columns)
    importance_df = importance_df[importance_df["feature_name"].isin(mordred_cols)]
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    importance_df.to_csv(
        output_dir / "global_mordred_multiclass_permutation_importance_all_features.csv",
        index=False,
    )

    top_df = importance_df.head(top_features).reset_index(drop=True)
    top_df.to_csv(
        output_dir / "global_mordred_multiclass_permutation_importance_top_features.csv",
        index=False,
    )
    metrics = {
        "sample_count": int(len(y_mordred)),
        "feature_count": int(x_mordred.shape[1]),
        "training_accuracy": float(acc),
        "training_balanced_accuracy": float(balanced_acc),
        "permutation_scoring": scoring,
        "surrogate_model": "RandomForestClassifier",
    }
    with open(output_dir / "global_mordred_multiclass_surrogate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return top_df, metrics


def plot_horizontal_importance(
    summary_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    xlabel: str,
    color: str = "#2C6E91",
    xerr_col: Optional[str] = None,
) -> None:
    plot_df = summary_df.copy()
    plot_df = plot_df.sort_values(x_col, ascending=True)

    fig_height = max(5.0, 0.42 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.barh(
        plot_df[y_col].astype(str),
        plot_df[x_col].astype(float),
        xerr=plot_df[xerr_col].astype(float) if xerr_col and xerr_col in plot_df else None,
        color=color,
        alpha=0.9,
        ecolor="#444444",
        capsize=2.5 if xerr_col and xerr_col in plot_df else 0.0,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_model_feature_mordred_correlation(
    feature_df: pd.DataFrame,
    top_features: pd.DataFrame,
    mordred_desc: pd.DataFrame,
    labels_for_features: pd.Series,
    aligned_labels: pd.Series,
    aligned_labeled_mask: pd.Series,
    output_dir: Path,
    top_mordred_descriptors: int,
    labeled_only: bool,
) -> pd.DataFrame:
    row_mask = make_model_row_mask(
        feature_df=feature_df,
        labels_for_features=labels_for_features,
        aligned_labels=aligned_labels,
        aligned_labeled_mask=aligned_labeled_mask,
        labeled_only=labeled_only,
    )

    selected_feature_names = top_features["feature_name"].astype(str).tolist()
    feat_df = feature_df.loc[row_mask, selected_feature_names].reset_index(drop=True)
    desc_df = mordred_desc.loc[row_mask].reset_index(drop=True)

    corr = pd.DataFrame(index=selected_feature_names, columns=desc_df.columns, dtype=float)
    for feature_name in selected_feature_names:
        corr.loc[feature_name] = desc_df.corrwith(feat_df[feature_name], method="pearson")

    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    descriptor_rank = corr.abs().max(axis=0).sort_values(ascending=False)
    selected_descriptors = descriptor_rank.head(top_mordred_descriptors).index.tolist()
    heatmap_data = corr.loc[:, selected_descriptors]

    heatmap_data.to_csv(output_dir / "pca_feature_mordred_heatmap_data.csv", index_label="model_feature")
    heatmap_long = (
        heatmap_data.reset_index()
        .rename(columns={"index": "model_feature"})
        .melt(id_vars="model_feature", var_name="Mordred_descriptor", value_name="Pearson_r")
    )
    heatmap_long.to_csv(output_dir / "pca_feature_mordred_heatmap_long.csv", index=False)
    descriptor_rank.to_csv(
        output_dir / "mordred_descriptor_max_abs_correlation_rank_to_pca_features.csv",
        header=["max_abs_r"],
    )
    return heatmap_data


def plot_pca_mordred_heatmap(heatmap_data: pd.DataFrame, output_dir: Path) -> None:
    width = max(11.0, 0.58 * heatmap_data.shape[1] + 3.5)
    height = max(6.5, 0.42 * heatmap_data.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_xlabel("Mordred descriptor")
    ax.set_ylabel("Important PCA/model feature")
    ax.set_title("Important PCA/model features mapped to Mordred descriptors")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_feature_mordred_correlation_heatmap.png", dpi=300)
    plt.close(fig)


def save_mordred_surrogate_outputs(
    shap_values: np.ndarray,
    mordred_desc: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    top_shap_features: int,
) -> None:
    shap_wide = pd.DataFrame(
        shap_values,
        columns=[f"SHAP__{col}" for col in mordred_desc.columns],
        index=mordred_desc.index,
    )
    feature_wide = mordred_desc.add_prefix("VALUE__")
    pd.concat([feature_wide, shap_wide], axis=1).to_csv(
        output_dir / "mordred_shap_values_wide.csv",
        index_label="sample_id",
    )

    top_bar = summary_df.head(top_shap_features).copy()
    plot_horizontal_importance(
        summary_df=top_bar,
        x_col="mean_abs_shap",
        y_col="feature",
        output_path=output_dir / "mordred_shap_importance_bar.png",
        title="Mordred descriptor importance for model target score",
        xlabel="Mean absolute SHAP value",
        color="#A65E2E",
    )


def run_mordred_shap_for_target(
    model: Any,
    feature_df: pd.DataFrame,
    mordred_desc: pd.DataFrame,
    formulation_features: pd.DataFrame,
    target_value: Any,
    output_dir: Path,
    random_state: int,
    n_estimators: int,
    n_jobs: int,
    top_shap_features: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_scores, target_metadata = get_target_scores(model, feature_df, target_value)
    score_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(target_scores)),
            "target_class": str(target_value),
            "target_score": target_scores,
        }
    )
    score_df.to_csv(output_dir / "model_target_class_scores.csv", index=False)

    x_combined = pd.concat([mordred_desc.reset_index(drop=True), formulation_features.reset_index(drop=True)], axis=1)

    surrogate = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=2,
        max_features=1.0,
    )
    surrogate.fit(x_combined, target_scores)
    fitted = surrogate.predict(x_combined)
    r2 = r2_score(target_scores, fitted)
    spearman = pd.Series(target_scores).corr(pd.Series(fitted), method="spearman")

    pd.DataFrame(
        {
            "sample_id": np.arange(len(target_scores)),
            "target_score": target_scores,
            "surrogate_fitted_score": fitted,
            "residual": target_scores - fitted,
        }
    ).to_csv(output_dir / "mordred_surrogate_target_scores.csv", index=False)

    shap_values_full = compute_tree_shap_values(surrogate, x_combined)
    
    mordred_feature_count = mordred_desc.shape[1]
    shap_values = shap_values_full[:, :mordred_feature_count]
    
    summary_df = summarize_shap_importance(shap_values, mordred_desc, target_value)

    summary_df.to_csv(output_dir / "mordred_shap_feature_importance_summary.csv", index=False)
    save_mordred_surrogate_outputs(
        shap_values=shap_values,
        mordred_desc=mordred_desc,
        summary_df=summary_df,
        output_dir=output_dir,
        top_shap_features=top_shap_features,
    )

    target_metadata.update(
        {
            "mordred_surrogate_model": "RandomForestRegressor",
            "mordred_surrogate_r2": None if pd.isna(r2) else float(r2),
            "mordred_surrogate_spearman": None if pd.isna(spearman) else float(spearman),
            "mordred_feature_count": int(mordred_desc.shape[1]),
        }
    )
    LOGGER.info(
        "Target %r Mordred-surrogate fidelity: R2=%.4f, Spearman=%.4f",
        target_value,
        float(r2),
        float(spearman) if not pd.isna(spearman) else float("nan"),
    )
    return target_metadata, summary_df


def main() -> None:
    args = parse_args()
    configure_logging(args.output_dir)
    ensure_optional_dependencies()

    label_mapping = parse_label_mapping(args.label_mapping)
    x_raw, df_work, smiles, labels, labeled_mask = load_and_align_inputs(
        embedding_path=args.embedding_path,
        csv_path=args.csv_path,
        smiles_col=args.smiles_col,
        label_col=args.label_col,
        label_mapping=label_mapping,
        missing_label_value=args.missing_label_value,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df_work.to_csv(args.output_dir / "aligned_metadata.csv", index_label="sample_id")
    pd.DataFrame(
        {
            "sample_id": np.arange(len(labels)),
            "raw_label": df_work[args.label_col],
            "model_label": labels,
            "is_labeled": labeled_mask,
            "SMILES": smiles,
        }
    ).to_csv(args.output_dir / "aligned_labels_smiles.csv", index=False)

    mordred_desc = load_mordred_descriptors_with_optional_external_cache(
        smiles=smiles,
        output_dir=args.output_dir,
        missing_threshold=args.mordred_missing_threshold,
        variance_threshold=args.variance_threshold,
        n_jobs=args.n_jobs,
        force_recompute=args.force_recompute_mordred,
        cache_dir=args.mordred_cache_dir,
    )

    aligned_df_for_bridge = df_work.reset_index(drop=True)
    aligned_smiles_for_bridge = smiles.reset_index(drop=True)
    aligned_labels_for_bridge = labels.reset_index(drop=True)
    aligned_labeled_mask_for_bridge = labeled_mask.reset_index(drop=True)
    mordred_desc_for_bridge = mordred_desc.reset_index(drop=True)
    training_sample_ids: Optional[pd.Series] = None
    if args.train_sample_ids_csv is not None:
        LOGGER.info("Loading training sample ids from %s", args.train_sample_ids_csv)
        training_sample_ids = load_training_sample_ids(args.train_sample_ids_csv)
        (
            aligned_df_for_bridge,
            aligned_smiles_for_bridge,
            aligned_labels_for_bridge,
            aligned_labeled_mask_for_bridge,
            mordred_desc_for_bridge,
        ) = subset_aligned_rows_by_sample_ids(
            sample_ids=training_sample_ids,
            df_work=df_work,
            smiles=smiles,
            labels=labels,
            labeled_mask=labeled_mask,
            mordred_desc=mordred_desc,
        )
        aligned_df_for_bridge.to_csv(
            args.output_dir / "aligned_training_subset_metadata.csv",
            index_label="subset_row",
        )
        pd.DataFrame(
            {
                "subset_row": np.arange(len(training_sample_ids)),
                "sample_id": training_sample_ids,
                "raw_label": aligned_df_for_bridge[args.label_col].reset_index(drop=True),
                "model_label": aligned_labels_for_bridge,
                "is_labeled": aligned_labeled_mask_for_bridge,
                "SMILES": aligned_smiles_for_bridge,
            }
        ).to_csv(args.output_dir / "aligned_training_subset_labels_smiles.csv", index=False)
        LOGGER.info(
            "Restricted raw/Mordred alignment space to %d training samples from %s",
            len(training_sample_ids),
            args.train_sample_ids_csv,
        )

    model, feature_df, labels_for_features, model_metadata = load_model_and_feature_space(
        model_path=args.model_path,
        require_model=args.require_model,
        x_raw=x_raw,
        labels=labels,
        labeled_mask=labeled_mask,
        random_state=args.random_state,
        n_estimators=args.proxy_n_estimators,
        n_jobs=args.n_jobs,
    )
    feature_df = feature_df.reset_index(drop=True)
    labels_for_features = pd.Series(labels_for_features).reset_index(drop=True)

    ratio_cols = [col for col in feature_df.columns if col.startswith("Ratio__")]
    ratio_df = feature_df[ratio_cols].reset_index(drop=True)
    
    cat_cols = ["Helper lipid", "Cholesterol", "PEG", "Additional component"]
    available_cat_cols = [c for c in cat_cols if c in aligned_df_for_bridge.columns]
    cat_df = aligned_df_for_bridge[available_cat_cols].fillna("Missing").astype(str)
    one_hot_df = pd.get_dummies(cat_df, prefix=available_cat_cols)
    
    formulation_features = pd.concat([ratio_df, one_hot_df.reset_index(drop=True)], axis=1)

    feature_df.to_csv(args.output_dir / "model_feature_space_matrix.csv", index_label="sample_id")
    pd.DataFrame(
        {
            "feature_index": np.arange(feature_df.shape[1]),
            "feature_name": list(feature_df.columns),
        }
    ).to_csv(args.output_dir / "model_feature_space_columns.csv", index=False)

    alignment_compatible = len(feature_df) == len(mordred_desc_for_bridge)
    if alignment_compatible:
        LOGGER.info(
            "Model feature rows (%d) match aligned raw rows (%d); PCA-to-Mordred bridge analyses will run.",
            len(feature_df),
            len(mordred_desc_for_bridge),
        )
    else:
        LOGGER.warning(
            "Model feature rows (%d) do not match aligned raw rows (%d). "
            "PCA-to-Mordred bridge analyses require row-wise alignment and will be skipped.",
            len(feature_df),
            len(mordred_desc_for_bridge),
        )

    top_features = run_model_space_permutation_importance(
        model=model,
        feature_df=feature_df,
        labels_for_features=labels_for_features,
        output_dir=args.output_dir,
        scoring=args.permutation_scoring,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        top_model_features=args.top_model_features,
    )
    plot_horizontal_importance(
        summary_df=top_features,
        x_col="importance_mean",
        y_col="feature_name",
        output_path=args.output_dir / "model_space_permutation_importance_bar.png",
        title="Permutation importance in PCA/model feature space",
        xlabel="Mean permutation importance",
        color="#2E6F40",
        xerr_col="importance_std",
    )

    global_mordred_importance_available = alignment_compatible
    global_mordred_metrics: Optional[Dict[str, Any]] = None
    if alignment_compatible:
        global_mordred_top, global_mordred_metrics = run_global_mordred_multiclass_importance(
            mordred_desc=mordred_desc_for_bridge,
            formulation_features=formulation_features,
            labels=aligned_labels_for_bridge,
            output_dir=args.output_dir,
            scoring=args.permutation_scoring,
            n_repeats=args.n_repeats,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            n_estimators=args.proxy_n_estimators,
            top_features=args.top_shap_features,
        )
        plot_horizontal_importance(
            summary_df=global_mordred_top,
            x_col="importance_mean",
            y_col="feature_name",
            output_path=args.output_dir / "global_mordred_multiclass_importance_bar.png",
            title="Overall multiclass Mordred importance",
            xlabel="Mean permutation importance",
            color="#8C5A2A",
            xerr_col="importance_std",
        )

    heatmap_output_available = False
    if alignment_compatible:
        heatmap_data = compute_model_feature_mordred_correlation(
            feature_df=feature_df,
            top_features=top_features,
            mordred_desc=mordred_desc_for_bridge,
            labels_for_features=labels_for_features,
            aligned_labels=aligned_labels_for_bridge,
            aligned_labeled_mask=aligned_labeled_mask_for_bridge,
            output_dir=args.output_dir,
            top_mordred_descriptors=args.top_mordred_descriptors,
            labeled_only=args.correlation_labeled_only,
        )
        plot_pca_mordred_heatmap(heatmap_data, args.output_dir)
        heatmap_output_available = True

    target_values, run_all_targets = resolve_target_values(
        target_class=args.target_class,
        labels=aligned_labels_for_bridge,
        labeled_mask=aligned_labeled_mask_for_bridge,
        label_mapping=label_mapping,
    )

    per_target_metadata: Dict[str, Dict[str, Any]] = {}
    per_target_outputs: Dict[str, str] = {}
    shap_importance_frames: List[pd.DataFrame] = []
    mordred_shap_available = alignment_compatible
    if alignment_compatible:
        LOGGER.info(
            "Running Mordred-surrogate SHAP for %d target class(es): %s",
            len(target_values),
            [str(target) for target in target_values],
        )
        for target_value in target_values:
            target_output_dir = (
                args.output_dir / f"target_class_{safe_path_token(target_value)}"
                if run_all_targets
                else args.output_dir
            )
            target_metadata, shap_importance = run_mordred_shap_for_target(
                model=model,
                feature_df=feature_df,
                mordred_desc=mordred_desc_for_bridge,
                formulation_features=formulation_features,
                target_value=target_value,
                output_dir=target_output_dir,
                random_state=args.random_state,
                n_estimators=args.proxy_n_estimators,
                n_jobs=args.n_jobs,
                top_shap_features=args.top_shap_features,
            )
            per_target_metadata[str(target_value)] = target_metadata
            per_target_outputs[str(target_value)] = str(target_output_dir.relative_to(args.output_dir))
            shap_importance_frames.append(shap_importance)

        all_shap_importance = pd.concat(shap_importance_frames, ignore_index=True)
        all_shap_importance.to_csv(
            args.output_dir / "all_target_mordred_shap_feature_importance_summary.csv",
            index=False,
        )
        shap_importance_wide = all_shap_importance.pivot(
            index="feature",
            columns="target_class",
            values="mean_abs_shap",
        ).reset_index()
        shap_importance_wide.to_csv(
            args.output_dir / "all_target_mordred_shap_mean_abs_wide.csv",
            index=False,
        )
    else:
        LOGGER.warning(
            "Skipping Mordred SHAP because the model feature matrix cannot be row-aligned to the current raw inputs."
        )

    metadata = {
        "embedding_path": str(args.embedding_path),
        "csv_path": str(args.csv_path),
        "model_path": str(args.model_path) if args.model_path else None,
        "output_dir": str(args.output_dir),
        "embedding_shape": list(x_raw.shape),
        "aligned_rows": int(len(df_work)),
        "labeled_rows": int(labeled_mask.sum()),
        "mordred_cleaned_shape": list(mordred_desc.shape),
        "bridge_aligned_rows": int(len(mordred_desc_for_bridge)),
        "model_feature_shape": list(feature_df.shape),
        "alignment_compatible": alignment_compatible,
        "train_sample_ids_csv": str(args.train_sample_ids_csv) if args.train_sample_ids_csv else None,
        "row_alignment_assumption": (
            "When alignment_compatible is true, the script assumes model feature rows and current raw rows "
            "refer to the same samples in the same order."
        ),
        "bridge_analysis_status": "ran" if alignment_compatible else "skipped_due_to_row_mismatch",
        "run_all_targets": run_all_targets,
        "target_class_argument": args.target_class,
        "target_classes_analyzed": [str(target) for target in target_values],
        "label_mapping": label_mapping,
        "missing_label_value": args.missing_label_value,
        "model_metadata": model_metadata,
        "global_mordred_multiclass_metrics": global_mordred_metrics,
        "per_target_metadata": per_target_metadata if mordred_shap_available else None,
        "per_target_output_dirs": per_target_outputs if mordred_shap_available else None,
        "outputs": {
            "model_feature_space_columns": "model_feature_space_columns.csv",
            "model_feature_space_matrix": "model_feature_space_matrix.csv",
            "model_space_permutation_importance_all": "model_space_permutation_importance_all_features.csv",
            "model_space_permutation_importance_top": "model_space_permutation_importance_top_features.csv",
            "model_space_permutation_importance_bar": "model_space_permutation_importance_bar.png",
            "global_mordred_multiclass_importance_all": (
                "global_mordred_multiclass_permutation_importance_all_features.csv"
                if global_mordred_importance_available
                else None
            ),
            "global_mordred_multiclass_importance_top": (
                "global_mordred_multiclass_permutation_importance_top_features.csv"
                if global_mordred_importance_available
                else None
            ),
            "global_mordred_multiclass_importance_bar": (
                "global_mordred_multiclass_importance_bar.png"
                if global_mordred_importance_available
                else None
            ),
            "global_mordred_multiclass_surrogate_metrics": (
                "global_mordred_multiclass_surrogate_metrics.json"
                if global_mordred_importance_available
                else None
            ),
            "pca_mordred_heatmap_matrix": (
                "pca_feature_mordred_heatmap_data.csv" if heatmap_output_available else None
            ),
            "pca_mordred_heatmap_long": (
                "pca_feature_mordred_heatmap_long.csv" if heatmap_output_available else None
            ),
            "pca_mordred_heatmap_png": (
                "pca_feature_mordred_correlation_heatmap.png" if heatmap_output_available else None
            ),
            "all_target_mordred_shap_feature_importance_summary": (
                "all_target_mordred_shap_feature_importance_summary.csv"
                if mordred_shap_available
                else None
            ),
            "all_target_mordred_shap_mean_abs_wide": (
                "all_target_mordred_shap_mean_abs_wide.csv" if mordred_shap_available else None
            ),
        },
    }
    with open(args.output_dir / "analysis_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    LOGGER.info("PCA-aware analysis complete. Outputs saved to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
