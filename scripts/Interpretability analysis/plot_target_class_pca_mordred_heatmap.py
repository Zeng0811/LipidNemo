#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot a PCA/model-feature vs Mordred correlation heatmap using the top descriptors
from a specific target_class SHAP summary.

Unlike the original PCA heatmap, which selects Mordred descriptors by global
maximum absolute correlation to the important PCA/model features, this script
uses the top-N descriptors from:

    target_class_{target}/mordred_shap_feature_importance_summary.csv

Those target-specific top descriptors become the heatmap x-axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analyze_lipidnemo_interpretability import safe_path_token


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build a PCA/model-feature vs Mordred correlation heatmap whose x-axis "
            "uses the top descriptors from one target_class SHAP summary."
        )
    )
    parser.add_argument(
        "--analysis_dir",
        type=Path,
        default=script_dir / "PCA_embedding-analysis",
        help="Existing PCA analysis output directory.",
    )
    parser.add_argument(
        "--target_class",
        required=True,
        help="Target class token used in target_class_{token} directory, for example 0/1/2/3.",
    )
    parser.add_argument(
        "--top_target_features",
        type=int,
        default=15,
        help="Number of target_class Mordred SHAP features to use as heatmap x-axis.",
    )
    parser.add_argument(
        "--top_model_features",
        type=int,
        default=20,
        help="Number of top PCA/model features to keep as heatmap y-axis.",
    )
    parser.add_argument(
        "--correlation_method",
        default="pearson",
        choices=["pearson", "spearman", "kendall"],
        help="Correlation method used between model features and Mordred descriptors.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Optional custom output file prefix. Defaults to target_class_{token}_top{N}.",
    )
    return parser.parse_args()


def load_model_feature_matrix(analysis_dir: Path) -> pd.DataFrame:
    path = analysis_dir / "model_feature_space_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Model feature matrix not found: {path}")
    return pd.read_csv(path, index_col=0).reset_index(drop=True)


def load_top_model_features(analysis_dir: Path, top_n: int) -> List[str]:
    path = analysis_dir / "model_space_permutation_importance_top_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Top model feature summary not found: {path}")

    df = pd.read_csv(path)
    if "feature_name" not in df.columns:
        raise KeyError(f"'feature_name' column not found in {path}")
    return df["feature_name"].astype(str).head(top_n).tolist()


def resolve_target_dir(analysis_dir: Path, target_class: str) -> Path:
    token = safe_path_token(target_class)
    path = analysis_dir / f"target_class_{token}"
    if not path.exists():
        raise FileNotFoundError(
            f"Target-class directory not found: {path}. "
            f"Check --target_class and existing analysis outputs."
        )
    return path


def load_top_target_descriptors(target_dir: Path, top_n: int) -> List[str]:
    path = target_dir / "mordred_shap_feature_importance_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Target SHAP summary not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"feature", "mean_abs_shap"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")

    top_df = (
        df.sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .drop_duplicates(subset=["feature"])
        .head(top_n)
    )
    return top_df["feature"].astype(str).tolist()


def load_aligned_mordred_subset(analysis_dir: Path, expected_rows: int) -> pd.DataFrame:
    mordred_path = analysis_dir / "mordred_cleaned_descriptors.csv"
    if not mordred_path.exists():
        raise FileNotFoundError(f"Mordred cleaned descriptor file not found: {mordred_path}")

    mordred_df = pd.read_csv(mordred_path, index_col=0)

    subset_path = analysis_dir / "aligned_training_subset_labels_smiles.csv"
    if subset_path.exists():
        subset_df = pd.read_csv(subset_path)
        if "sample_id" not in subset_df.columns:
            raise KeyError(f"'sample_id' column not found in {subset_path}")
        sample_ids = pd.to_numeric(subset_df["sample_id"], errors="raise").astype(int).to_numpy()
        if sample_ids.min() < 0 or sample_ids.max() >= len(mordred_df):
            raise ValueError(
                "sample_id values in aligned_training_subset_labels_smiles.csv are out of range for "
                f"{mordred_path}"
            )
        mordred_subset = mordred_df.iloc[sample_ids].reset_index(drop=True)
    else:
        mordred_subset = mordred_df.reset_index(drop=True)

    if len(mordred_subset) != expected_rows:
        raise ValueError(
            "Row mismatch after aligning Mordred descriptors to model feature space: "
            f"mordred_rows={len(mordred_subset)}, model_feature_rows={expected_rows}."
        )
    return mordred_subset


def compute_target_heatmap(
    feature_df: pd.DataFrame,
    mordred_df: pd.DataFrame,
    selected_model_features: List[str],
    selected_descriptors: List[str],
    correlation_method: str,
) -> pd.DataFrame:
    missing_model = [col for col in selected_model_features if col not in feature_df.columns]
    if missing_model:
        raise KeyError(f"Selected model features are missing from model_feature_space_matrix.csv: {missing_model}")

    missing_desc = [col for col in selected_descriptors if col not in mordred_df.columns]
    if missing_desc:
        raise KeyError(f"Selected Mordred descriptors are missing from mordred_cleaned_descriptors.csv: {missing_desc}")

    feat_df = feature_df.loc[:, selected_model_features].reset_index(drop=True)
    desc_df = mordred_df.loc[:, selected_descriptors].reset_index(drop=True)

    corr = pd.DataFrame(index=selected_model_features, columns=selected_descriptors, dtype=float)
    for feature_name in selected_model_features:
        corr.loc[feature_name] = desc_df.corrwith(feat_df[feature_name], method=correlation_method)

    return corr.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def plot_heatmap(
    heatmap_data: pd.DataFrame,
    output_png: Path,
    title: str,
    correlation_method: str,
) -> None:
    label_map = {
        "pearson": "Pearson r",
        "spearman": "Spearman rho",
        "kendall": "Kendall tau",
    }
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
        cbar_kws={"label": label_map.get(correlation_method, "Correlation")},
        ax=ax,
    )
    ax.set_xlabel("Target-class top Mordred descriptor")
    ax.set_ylabel("Important PCA/model feature")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    target_dir = resolve_target_dir(analysis_dir, str(args.target_class))
    target_token = safe_path_token(args.target_class)

    feature_df = load_model_feature_matrix(analysis_dir)
    selected_model_features = load_top_model_features(analysis_dir, args.top_model_features)
    selected_descriptors = load_top_target_descriptors(target_dir, args.top_target_features)
    mordred_df = load_aligned_mordred_subset(analysis_dir, expected_rows=len(feature_df))

    heatmap_data = compute_target_heatmap(
        feature_df=feature_df,
        mordred_df=mordred_df,
        selected_model_features=selected_model_features,
        selected_descriptors=selected_descriptors,
        correlation_method=args.correlation_method,
    )

    prefix = args.output_prefix or f"target_class_{target_token}_top{args.top_target_features}"
    matrix_path = analysis_dir / f"{prefix}_pca_feature_mordred_heatmap_data.csv"
    long_path = analysis_dir / f"{prefix}_pca_feature_mordred_heatmap_long.csv"
    png_path = analysis_dir / f"{prefix}_pca_feature_mordred_correlation_heatmap.png"
    metadata_path = analysis_dir / f"{prefix}_pca_feature_mordred_heatmap_metadata.json"

    heatmap_data.to_csv(matrix_path, index_label="model_feature")
    heatmap_long = (
        heatmap_data.reset_index()
        .rename(columns={"index": "model_feature"})
        .melt(id_vars="model_feature", var_name="Mordred_descriptor", value_name="Correlation")
    )
    heatmap_long.to_csv(long_path, index=False)

    title = (
        f"Important PCA/model features vs target_class {args.target_class} "
        f"top-{args.top_target_features} Mordred descriptors ({args.correlation_method})"
    )
    plot_heatmap(heatmap_data, png_path, title, args.correlation_method)

    metadata: dict[str, Any] = {
        "analysis_dir": str(analysis_dir),
        "target_class": str(args.target_class),
        "target_dir": str(target_dir),
        "top_target_features": int(args.top_target_features),
        "top_model_features": int(args.top_model_features),
        "correlation_method": args.correlation_method,
        "selected_model_features": selected_model_features,
        "selected_target_descriptors": selected_descriptors,
        "outputs": {
            "heatmap_matrix": matrix_path.name,
            "heatmap_long": long_path.name,
            "heatmap_png": png_path.name,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
