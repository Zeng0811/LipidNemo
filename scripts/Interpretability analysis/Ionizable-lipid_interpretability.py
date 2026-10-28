#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LipidNemo Interpretability Core Utilities Module.

This script contains standard adapter classes, data loading/alignment helpers,
Mordred descriptor calculation & caching pipelines, surrogate tree model SHAP wrappers,
and formatting utilities required by PCA-aware interpretability analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score


LOGGER = logging.getLogger("LipidNemoInterpretability")

COMPONENT_NAMES = [
    "Ionizable_lipid",
    "Helper_lipid",
    "Cholesterol",
    "PEG",
    "Additional_component",
]

MORDRED_CACHE_SCHEMA_VERSION = 1
MORDRED_CLEANED_CACHE_FILE = "mordred_cleaned_descriptors.csv"
MORDRED_CACHE_METADATA_FILE = "mordred_cache_metadata.json"


class AutoGluonPredictorWrapper:
    """Small sklearn-like adapter for AutoGluon TabularPredictor.

    sklearn.inspection.permutation_importance expects an estimator with
    predict/predict_proba methods that accepts the same X object supplied to it.
    AutoGluon normally expects a pandas DataFrame with the exact training column
    names, so this wrapper converts the numpy embedding matrix back to that
    DataFrame layout before prediction.
    """

    def __init__(
        self,
        predictor: Any,
        feature_columns: Sequence[Any],
        training_X: Optional[pd.DataFrame] = None,
        training_y: Optional[pd.Series] = None,
    ):
        self.predictor = predictor
        self.feature_columns = list(feature_columns)
        self.training_X = training_X
        self.training_y = training_y
        class_labels = getattr(predictor, "class_labels", None)
        if class_labels is not None:
            self.classes_ = np.asarray(class_labels, dtype=object)

    def _to_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        X_array = np.asarray(X)
        if X_array.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X_array.shape}")
        if X_array.shape[1] != len(self.feature_columns):
            raise ValueError(
                "AutoGluon predictor feature count mismatch: "
                f"input has {X_array.shape[1]} columns, model expects {len(self.feature_columns)}."
            )
        return pd.DataFrame(X_array, columns=self.feature_columns)

    def fit(self, X: Any, y: Any = None) -> "AutoGluonPredictorWrapper":
        """No-op fit method for sklearn inspection utilities."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        pred = self.predictor.predict(self._to_dataframe(X), as_pandas=False)
        return np.asarray(pred)

    def predict_proba(self, X: Any) -> np.ndarray:
        proba = self.predictor.predict_proba(self._to_dataframe(X), as_pandas=True)
        if hasattr(proba, "columns"):
            self.classes_ = np.asarray(proba.columns, dtype=object)
            return proba.to_numpy()
        return np.asarray(proba)


def safe_path_token(value: Any) -> str:
    """Sanitize string values for safe path tokens."""
    s = str(value).strip()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s if s else "blank"


def make_full_feature_names() -> List[str]:
    """Generate default column names for LipidNemo input feature spaces."""
    feature_names = [f"Ratio__{name}" for name in COMPONENT_NAMES]
    feature_names.extend([f"Embedding__{i}" for i in range(2560)])
    return feature_names


def configure_logging(output_dir: Path) -> None:
    """Configure file and stdout logging for interpretability analyses."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def ensure_optional_dependencies() -> None:
    """Check for optional domain packages and log missing warnings."""
    missing = []
    try:
        import rdkit
    except ImportError:
        missing.append("rdkit")
    try:
        import mordred
    except ImportError:
        missing.append("mordred")
    try:
        import shap
    except ImportError:
        missing.append("shap")
    if missing:
        LOGGER.warning("Missing optional dependencies: %s", ", ".join(missing))


def parse_label_mapping(mapping_text: str) -> Dict[str, Any]:
    """Parse comma-separated label mapping string like 'Liver:0,Spleen:1,Lung:2'."""
    if not mapping_text or not mapping_text.strip():
        return {}
    mapping = {}
    for part in mapping_text.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            k, v = k.strip(), v.strip()
            if v.isdigit():
                mapping[k] = int(v)
            else:
                mapping[k] = v
    return mapping


def resolve_target_values(
    target_class: str,
    labels: pd.Series,
    labeled_mask: pd.Series,
    label_mapping: Dict[str, Any],
) -> Tuple[List[Any], bool]:
    """Resolve target class specification to actual class label values."""
    if target_class.lower() == "all":
        unique_targets = list(pd.Series(labels[labeled_mask]).unique())
        return unique_targets, True

    target_val = target_class
    if label_mapping:
        if target_class in label_mapping:
            target_val = label_mapping[target_class]
        else:
            try:
                target_val = int(target_class)
            except ValueError:
                pass
    else:
        try:
            target_val = int(target_class)
        except ValueError:
            pass
    return [target_val], False


def save_mordred_cache_metadata(
    smiles: pd.Series,
    output_dir: Path,
    missing_threshold: float,
    variance_threshold: float,
    descriptor_columns: Sequence[str],
) -> None:
    """Save metadata JSON file for cached Mordred descriptor calculations."""
    smiles_clean = smiles.dropna().astype(str).tolist()
    smiles_hash = hashlib.sha256("".join(sorted(smiles_clean)).encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": MORDRED_CACHE_SCHEMA_VERSION,
        "smiles_count": len(smiles),
        "smiles_hash": smiles_hash,
        "missing_threshold": missing_threshold,
        "variance_threshold": variance_threshold,
        "n_descriptors": len(descriptor_columns),
        "descriptor_columns": list(descriptor_columns),
    }
    meta_path = output_dir / MORDRED_CACHE_METADATA_FILE
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_cached_mordred_descriptors(
    smiles: pd.Series,
    output_dir: Path,
    missing_threshold: float,
    variance_threshold: float,
) -> Optional[pd.DataFrame]:
    """Load previously calculated Mordred descriptors if cache metadata matches."""
    cache_csv = output_dir / MORDRED_CLEANED_CACHE_FILE
    meta_json = output_dir / MORDRED_CACHE_METADATA_FILE
    if not cache_csv.exists() or not meta_json.exists():
        return None
    try:
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("schema_version") != MORDRED_CACHE_SCHEMA_VERSION:
            return None
        if abs(meta.get("missing_threshold", 0) - missing_threshold) > 1e-6:
            return None
        if abs(meta.get("variance_threshold", 0) - variance_threshold) > 1e-12:
            return None
        smiles_clean = smiles.dropna().astype(str).tolist()
        smiles_hash = hashlib.sha256("".join(sorted(smiles_clean)).encode("utf-8")).hexdigest()
        if meta.get("smiles_hash") != smiles_hash:
            return None
        df = pd.read_csv(cache_csv)
        if len(df) != len(smiles):
            return None
        return df
    except Exception as e:
        LOGGER.warning("Error reading Mordred cache: %s", e)
        return None


def compute_and_clean_mordred_descriptors(
    smiles: pd.Series,
    output_dir: Path,
    missing_threshold: float = 0.1,
    variance_threshold: float = 1e-12,
    n_jobs: int = -1,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Compute and clean 2D Mordred molecular descriptors from SMILES."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not force_recompute:
        cached = load_cached_mordred_descriptors(
            smiles=smiles,
            output_dir=output_dir,
            missing_threshold=missing_threshold,
            variance_threshold=variance_threshold,
        )
        if cached is not None:
            LOGGER.info("Loaded cached Mordred descriptors: shape %s", cached.shape)
            return cached
    else:
        LOGGER.info("Ignoring Mordred descriptor cache because --force_recompute_mordred was set.")

    from mordred import Calculator, descriptors
    from rdkit import Chem

    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(str(s)) if pd.notna(s) else None for s in smiles]

    df_raw = calc.pandas(mols, nproc=n_jobs)
    df_num = df_raw.apply(pd.to_numeric, errors="coerce")

    missing_frac = df_num.isna().mean()
    keep_cols = missing_frac[missing_frac <= missing_threshold].index
    df_filtered = df_num[keep_cols]

    df_imputed = df_filtered.fillna(df_filtered.mean())

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df_imputed)
    cleaned_cols = df_imputed.columns[selector.get_support()]
    df_cleaned = df_imputed[cleaned_cols]

    df_cleaned.to_csv(output_dir / MORDRED_CLEANED_CACHE_FILE, index=False)
    save_mordred_cache_metadata(
        smiles=smiles,
        output_dir=output_dir,
        missing_threshold=missing_threshold,
        variance_threshold=variance_threshold,
        descriptor_columns=list(df_cleaned.columns),
    )
    LOGGER.info("Computed and cleaned Mordred descriptors: shape %s", df_cleaned.shape)
    return df_cleaned


def load_autogluon_predictor(
    model_dir: Path,
    fallback_feature_names: Sequence[str],
) -> AutoGluonPredictorWrapper:
    """Load AutoGluon TabularPredictor and return sklearn adapter wrapper."""
    from autogluon.tabular import TabularPredictor

    predictor = TabularPredictor.load(str(model_dir))
    feature_metadata = getattr(predictor, "feature_metadata", None)
    if feature_metadata is not None and hasattr(feature_metadata, "get_features"):
        feature_cols = feature_metadata.get_features()
    else:
        feature_cols = fallback_feature_names
    return AutoGluonPredictorWrapper(predictor=predictor, feature_columns=feature_cols)


def load_and_align_inputs(
    embedding_path: Path,
    csv_path: Path,
    smiles_col: str,
    label_col: str,
    label_mapping: Dict[str, Any],
    missing_label_value: str,
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load embeddings and CSV data, alignment check, and map class labels."""
    embeddings = np.load(embedding_path)
    df_csv = pd.read_csv(csv_path)

    if len(embeddings) != len(df_csv):
        raise ValueError(
            f"Row count mismatch: embedding has {len(embeddings)} rows, CSV has {len(df_csv)} rows."
        )

    smiles = df_csv[smiles_col]
    raw_labels = df_csv[label_col].copy()

    if missing_label_value != "":
        raw_labels = raw_labels.fillna(missing_label_value)

    if label_mapping:
        mapped_labels = raw_labels.map(label_mapping)
    else:
        mapped_labels = raw_labels

    labeled_mask = mapped_labels.notna()
    return embeddings, df_csv, smiles, mapped_labels, labeled_mask


def get_target_scores(
    model: Any,
    X: np.ndarray,
    target_value: Any,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract model target class score probabilities or predictions."""
    meta: Dict[str, Any] = {"target_value": target_value}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes_list = list(classes)
            if target_value in classes_list:
                idx = classes_list.index(target_value)
                scores = proba[:, idx]
                meta["score_type"] = "probability"
                meta["class_index"] = idx
                return np.asarray(scores), meta
            try:
                target_val_int = int(target_value)
                if target_val_int in classes_list:
                    idx = classes_list.index(target_val_int)
                    scores = proba[:, idx]
                    meta["score_type"] = "probability"
                    meta["class_index"] = idx
                    return np.asarray(scores), meta
            except Exception:
                pass
        scores = proba[:, 0]
        meta["score_type"] = "probability_fallback"
        return np.asarray(scores), meta

    preds = model.predict(X)
    scores = (np.asarray(preds) == target_value).astype(float)
    meta["score_type"] = "binary_indicator"
    return scores, meta


def compute_tree_shap_values(
    surrogate: RandomForestRegressor,
    macro_df: pd.DataFrame,
) -> np.ndarray:
    """Compute Tree SHAP values using a trained RandomForest surrogate."""
    import shap

    explainer = shap.TreeExplainer(surrogate)
    shap_values = explainer.shap_values(macro_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values)


def summarize_shap_importance(
    shap_values: np.ndarray,
    macro_df: pd.DataFrame,
    target_value: Any,
) -> pd.DataFrame:
    """Summarize mean absolute SHAP feature importances."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    summary_df = pd.DataFrame(
        {
            "feature": list(macro_df.columns),
            "mean_abs_shap": mean_abs_shap,
            "target_class": str(target_value),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary_df
