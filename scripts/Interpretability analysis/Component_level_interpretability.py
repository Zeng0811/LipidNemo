#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LipidNemo Component-Level Interpretability Pipeline
Enhanced Version: 5-Fold Cross-Validation Out-of-Fold (OOF) & Entity-Coupled Permutation

This script performs:
1. 5-Fold Stratified Cross-Validation with TabPFN:
   - Strictly out-of-sample evaluation across formulations (no data leakage).
   - Entity-coupled permutation feature importance (Structure + Molar % permuted as a unified unit).
   - Evaluates component importance using Accuracy Drop (delta_Acc).
   - Exports:
     - group_permutation_importance_detailed.csv
     - group_permutation_importance_summary.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("ComponentLevelInterp")

TARGET_CLASSES = ["Liver", "Lung", "Spleen", "None"]

# Entity-coupled groups: (embedding_range, molar_ratio_index)
# Raw feature vector: 0..2559 = embeddings (5 x 512D), 2560..2564 = molar ratios
ENTITY_GROUPS = {
    "Ionizable_lipid": ((0, 512), 2560),
    "Helper_lipid": ((512, 1024), 2561),
    "Cholesterol": ((1024, 1536), 2562),
    "PEG": ((1536, 2048), 2563),
    "Additional_component": ((2048, 2560), 2564),
    "Molar_ratios": (None, (2560, 2565)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LipidNemo 5-Fold OOF Component-Level Interpretability (Accuracy Drop)"
    )
    parser.add_argument(
        "--data_csv",
        type=Path,
        default=Path("data/LNP-447.csv"),
        help="Path to LNP metadata CSV file.",
    )
    parser.add_argument(
        "--embedding_path",
        type=Path,
        default=Path("data/LNP-447-lipidnemo.npy"),
        help="Path to raw (N, 2565) MegaMolBART embedding npy file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/component_level"),
        help="Output directory for generated CSVs.",
    )
    parser.add_argument(
        "--pca_n",
        type=int,
        default=20,
        help="Number of PCA components fitted on training embeddings.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=20,
        help="Number of random permutation repeats per group per fold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for cross-validation splitting and permutations.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for TabPFN inference ('cuda:0' or 'cpu').",
    )
    return parser.parse_args()


# =========================================================================
# Metric Computation
# =========================================================================

def compute_accuracy(y_true_indices: np.ndarray, probs: np.ndarray) -> float:
    """Calculate multiclass Accuracy safely."""
    y_pred_indices = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true_indices, y_pred_indices)
    return float(acc)


# =========================================================================
# 5-Fold Cross-Validation OOF Permutation Feature Importance
# =========================================================================

def run_5fold_oof_permutation_importance(
    X_raw: np.ndarray,
    y_idx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute 5-Fold CV OOF evaluation with Entity-Coupled Permutation (Accuracy Drop).
    Returns:
        df_detailed: all fold and repeat accuracy metrics and deltas
        df_summary: aggregated mean +/- std accuracy performance drops
    """
    from tabpfn import TabPFNClassifier

    LOGGER.info("=" * 75)
    LOGGER.info("[START] 5-Fold CV Out-of-Fold Permutation Importance (Accuracy Drop)")
    LOGGER.info(
        f"[INFO] Splits: {args.n_splits} | PCA Components: {args.pca_n} | Repeats: {args.n_repeats}"
    )
    LOGGER.info("=" * 75)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    detailed_records: List[Dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_idx)):
        LOGGER.info(
            f"[INFO] --- Processing Fold {fold + 1}/{args.n_splits} "
            f"(Train={len(train_idx)}, Test={len(test_idx)}) ---"
        )

        # 1. Feature Preprocessing (strictly fit on train fold)
        scaler = StandardScaler()
        X_train_emb_scaled = scaler.fit_transform(X_raw[train_idx, :2560])
        pca = PCA(n_components=args.pca_n, random_state=args.seed)
        X_train_pca = pca.fit_transform(X_train_emb_scaled)
        X_train_final = np.hstack([X_train_pca, X_raw[train_idx, 2560:]]).astype(np.float32)

        X_test_emb_scaled = scaler.transform(X_raw[test_idx, :2560])
        X_test_pca = pca.transform(X_test_emb_scaled)
        X_test_final = np.hstack([X_test_pca, X_raw[test_idx, 2560:]]).astype(np.float32)

        # 2. Train TabPFN Classifier
        clf = TabPFNClassifier(
            device=args.device,
            n_estimators=4,
            random_state=args.seed + fold,
            ignore_pretraining_limits=True,
        )
        clf.fit(X_train_final, y_idx[train_idx])

        # 3. Baseline Out-of-Fold Evaluation
        test_base_probs = clf.predict_proba(X_test_final)
        base_acc = compute_accuracy(y_idx[test_idx], test_base_probs)

        detailed_records.append({
            "fold": fold + 1,
            "component_group": "Baseline (Unpermuted)",
            "repeat": 0,
            "metric_accuracy": base_acc,
            "delta_accuracy": 0.0,
        })

        LOGGER.info(f"[INFO]     Fold {fold + 1} Baseline: Acc={base_acc:.4f}")

        # 4. Entity-Coupled Permutation Testing
        rng = np.random.RandomState(args.seed + fold * 100)
        N_test = len(test_idx)

        for group_name, ginfo in ENTITY_GROUPS.items():
            for r in range(args.n_repeats):
                X_test_perm = X_raw[test_idx].copy()
                perm = rng.permutation(N_test)

                if group_name == "Molar_ratios":
                    # Permute the entire 5-dimensional molar ratio vector
                    X_test_perm[:, 2560:2565] = X_test_perm[perm, 2560:2565]
                else:
                    emb_range, ratio_idx = ginfo
                    # Shuffling structure and molar ratio simultaneously as a single entity
                    X_test_perm[:, emb_range[0]:emb_range[1]] = X_test_perm[perm, emb_range[0]:emb_range[1]]
                    X_test_perm[:, ratio_idx] = X_test_perm[perm, ratio_idx]

                # Transform with training scaler and PCA
                X_p_emb_scaled = scaler.transform(X_test_perm[:, :2560])
                X_p_pca = pca.transform(X_p_emb_scaled)
                X_p_final = np.hstack([X_p_pca, X_test_perm[:, 2560:]]).astype(np.float32)

                p_probs = clf.predict_proba(X_p_final)
                p_acc = compute_accuracy(y_idx[test_idx], p_probs)

                detailed_records.append({
                    "fold": fold + 1,
                    "component_group": group_name,
                    "repeat": r + 1,
                    "metric_accuracy": p_acc,
                    "delta_accuracy": base_acc - p_acc,
                })

    df_detailed = pd.DataFrame(detailed_records)
    detailed_csv = args.output_dir / "group_permutation_importance_detailed.csv"
    df_detailed.to_csv(detailed_csv, index=False, encoding="utf-8-sig")
    LOGGER.info(f"[SAVE] Detailed permutation results exported to: {detailed_csv}")

    # Summary Statistics across all folds and repeats
    df_deltas = df_detailed[df_detailed["component_group"] != "Baseline (Unpermuted)"].copy()

    summary_list: List[Dict[str, Any]] = []
    for gname in ENTITY_GROUPS.keys():
        sub = df_deltas[df_deltas["component_group"] == gname]
        vals = sub["delta_accuracy"].dropna().values
        summary_list.append({
            "component_group": gname,
            "accuracy_drop_mean": float(np.mean(vals)) if len(vals) > 0 else 0.0,
            "accuracy_drop_std": float(np.std(vals)) if len(vals) > 0 else 0.0,
        })

    df_summary = pd.DataFrame(summary_list)
    summary_csv = args.output_dir / "group_permutation_importance_summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    LOGGER.info(f"[SAVE] Summary permutation statistics exported to: {summary_csv}")

    return df_detailed, df_summary


# =========================================================================
# Main Execution Pipeline
# =========================================================================

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("[INFO] Loading LNP dataset and raw molecular embeddings...")
    df = pd.read_csv(args.data_csv)
    X_raw = np.load(args.embedding_path)

    y_raw = df["Organ"].fillna("None").astype(str).str.strip().values
    class_to_idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    y_idx = np.array([class_to_idx.get(y, 3) for y in y_raw], dtype=int)

    LOGGER.info(f"[INFO] Dataset aligned: N={len(df)} samples | Features={X_raw.shape[1]}")
    for c in TARGET_CLASSES:
        LOGGER.info(f"[INFO]   Class {c:8s}: {(y_raw == c).sum()} samples")

    # 5-Fold OOF Permutation Feature Importance (Accuracy Drop only)
    df_detailed, df_summary = run_5fold_oof_permutation_importance(X_raw, y_idx, args)

    LOGGER.info("=" * 75)
    LOGGER.info("[DONE] COMPONENT-LEVEL INTERPRETABILITY PIPELINE COMPLETED")
    LOGGER.info(f"[INFO] Artifacts exported to: {args.output_dir}")
    LOGGER.info("   - Detailed CSV: group_permutation_importance_detailed.csv")
    LOGGER.info("   - Summary CSV: group_permutation_importance_summary.csv")
    LOGGER.info("=" * 75)


if __name__ == "__main__":
    main()
