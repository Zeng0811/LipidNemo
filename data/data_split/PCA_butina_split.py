import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_DATA_CSV = Path("data/LNP-447.csv")
DEFAULT_EMBEDDING_NPY = Path("data/LNP-447-ft-final.npy")

TARGET_COLUMN = "Organ"
TARGET_CLASSES = ("Liver", "Lung", "Spleen", "None")
IONIZABLE_SMILES_COLUMN = "Ionizable lipid(S)"


def data_load(file_path):
    """Load .npy / .npz / .csv data."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".npy":
        return np.load(file_path)
    if ext == ".npz":
        with np.load(file_path) as loaded:
            keys = list(loaded.keys())
            return loaded["arr_0"] if "arr_0" in keys else loaded[keys[0]]
    if ext == ".csv":
        return pd.read_csv(file_path).values.astype(np.float32)

    raise ValueError(f"[ERROR] Unsupported file format: {file_path.suffix}")


def _clean_target_series(series):
    return (
        series.fillna("None")
        .astype(str)
        .str.strip()
        .replace({"nan": "None", "NaN": "None"})
    )


def _load_aligned_lnp_dataset(
    data_csv,
    embedding_npy,
    target_column=TARGET_COLUMN,
    target_classes=TARGET_CLASSES,
):
    """
    Align CSV and embedding row counts, and filter target labels.

    Core behavior preserved:
    1. Truncate to minimum length if CSV and embedding row counts differ.
    2. Retain only target label classes for subsequent training and evaluation.
    """
    data_csv = Path(data_csv)
    embedding_npy = Path(embedding_npy)

    df = pd.read_csv(data_csv)
    x_all = data_load(embedding_npy)

    if x_all.ndim != 2:
        raise ValueError(f"[ERROR] Embedding must be a 2D array, current shape={x_all.shape}")

    aligned_rows = min(len(df), len(x_all))
    if len(df) != len(x_all):
        warnings.warn(
            "Row count mismatch between CSV and embedding, truncated to minimum length. "
            f"CSV={len(df)}, embedding={len(x_all)}, aligned={aligned_rows}"
        )

    df = df.iloc[:aligned_rows].copy()
    x_all = x_all[:aligned_rows]
    df["sample_id"] = np.arange(aligned_rows, dtype=int)

    df[target_column] = _clean_target_series(df[target_column])
    mask = df[target_column].isin(target_classes)

    df_filtered = df.loc[mask].copy().reset_index(drop=True)
    x_filtered = np.nan_to_num(
        x_all[mask.values],
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    ).astype(np.float32)
    y_filtered = df_filtered[target_column].to_numpy()

    if len(df_filtered) == 0:
        raise ValueError("[ERROR] No available samples after filtering target labels.")

    return df_filtered, x_filtered, y_filtered


def _subset_sum_cluster_split(
    group_sizes,
    target_size,
    seed=42,
    fixed_train_count=0,
    min_train_count=0,
):
    """
    Approximate target test set size using subset-sum DP without splitting clusters.

    Parameters
    ----------
    group_sizes : dict
        {group_id: sample_count}
    target_size : int
        Target number of test set samples.
    seed : int, default=42
        Random seed for shuffling cluster iteration order.
    fixed_train_count : int, default=0
        Number of samples forcibly placed in the training set (e.g., invalid SMILES).
    min_train_count : int, default=0
        Minimum number of samples retained in the training set (hard constraint to avoid over-shrinking).
    """
    items = [(group_id, int(size)) for group_id, size in group_sizes.items() if int(size) > 0]
    if not items:
        return set()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    items = [items[idx] for idx in order]

    total_assignable = sum(size for _, size in items)
    max_sum = total_assignable

    reachable = [False] * (max_sum + 1)
    predecessor = [None] * (max_sum + 1)
    reachable[0] = True

    for item_idx, (_, size) in enumerate(items):
        for current_sum in range(max_sum, size - 1, -1):
            if reachable[current_sum - size] and not reachable[current_sum]:
                reachable[current_sum] = True
                predecessor[current_sum] = (current_sum - size, item_idx)

    valid_sums = []
    # Prioritize candidate solutions satisfying min_train_count
    if min_train_count > 0:
        for current_sum in range(1, max_sum + 1):
            remaining_train = total_assignable - current_sum + fixed_train_count
            if reachable[current_sum] and remaining_train >= min_train_count:
                valid_sums.append(current_sum)

    # If unspecified or no solution satisfies min_train_count, fall back to remaining_train >= 1
    if not valid_sums:
        for current_sum in range(1, max_sum + 1):
            remaining_train = total_assignable - current_sum + fixed_train_count
            if reachable[current_sum] and remaining_train >= 1:
                valid_sums.append(current_sum)

    if not valid_sums:
        return set()

    best_sum = min(valid_sums, key=lambda value: (abs(value - target_size), value))

    chosen_indices = set()
    cursor = best_sum
    while cursor != 0:
        prev_sum, item_idx = predecessor[cursor]
        chosen_indices.add(item_idx)
        cursor = prev_sum

    return {items[item_idx][0] for item_idx in chosen_indices}


def _project_to_2d(features, seed=42, prefer_umap=True, metric="jaccard"):
    """Prioritize UMAP for 2D projection, fallback to PCA if unavailable. Default metric is jaccard (Tanimoto distance) for chemical fingerprints."""
    features = np.asarray(features)

    if prefer_umap:
        try:
            import umap

            input_features = features.astype(bool) if metric == "jaccard" else features.astype(np.float32)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(25, max(2, len(input_features) - 1)),
                min_dist=0.2,
                metric=metric,
                random_state=seed,
            )
            return reducer.fit_transform(input_features), f"UMAP({metric})"
        except Exception as exc:  # pragma: no cover - environment dependent
            warnings.warn(f"UMAP is unavailable, automatically falling back to PCA. Reason: {exc}")

    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(np.asarray(features, dtype=np.float32)), "PCA"


def _fingerprints_to_numpy(fingerprints, n_bits):
    from rdkit import DataStructs

    array = np.zeros((len(fingerprints), n_bits), dtype=np.float32)
    for idx, fp in enumerate(fingerprints):
        row = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, row)
        array[idx] = row
    return array


def _save_butina_projection(
    fingerprint_matrix,
    unique_splits,
    figure_path,
    seed,
    cutoff,
    projection_metric="jaccard",
):
    import matplotlib.pyplot as plt

    projection, method_name = _project_to_2d(
        fingerprint_matrix,
        seed=seed,
        prefer_umap=True,
        metric=projection_metric,
    )

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    style_map = {
        "train": {"color": "#1f77b4", "marker": "o", "label": "Train"},
        "val": {"color": "#2ca02c", "marker": "s", "label": "Val"},
        "test": {"color": "#d62728", "marker": "^", "label": "Test"},
    }

    for split_name, style in style_map.items():
        mask = unique_splits == split_name
        if not np.any(mask):
            continue

        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            c=style["color"],
            marker=style["marker"],
            s=52,
            alpha=0.85,
            label=f"{style['label']} unique lipids ({int(mask.sum())})",
        )

    plt.title(f"Butina OOD Split Projection ({method_name}, cutoff={cutoff})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path


def _save_split_metadata(
    df_filtered,
    target_column,
    ionizable_smiles_column,
    output_dir,
    radius,
    n_bits,
    cutoff,
    test_ratio,
    seed,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_columns = [
        "sample_id",
        target_column,
        ionizable_smiles_column,
        "_ionizable_canonical_smiles",
        "_cluster_id",
        "_split",
    ]
    existing_columns = [col for col in selected_columns if col in df_filtered.columns]

    all_samples = df_filtered.loc[:, existing_columns].copy()
    all_samples.to_csv(output_dir / "butina_split_all_samples.csv", index=False, encoding="utf-8-sig")

    train_df = all_samples.loc[all_samples["_split"].eq("train")].copy()
    test_df = all_samples.loc[all_samples["_split"].eq("test")].copy()

    train_df.to_csv(output_dir / "butina_train_samples.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "butina_test_samples.csv", index=False, encoding="utf-8-sig")

    train_df[["sample_id"]].to_csv(
        output_dir / "train_sample_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_df[["sample_id"]].to_csv(
        output_dir / "test_sample_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "sample_count_filtered": int(len(df_filtered)),
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "train_fraction": float(len(train_df) / len(df_filtered)),
        "test_fraction": float(len(test_df) / len(df_filtered)),
        "target_column": str(target_column),
        "ionizable_smiles_column": str(ionizable_smiles_column),
        "split_method": "butina",
        "split_params": {
            "radius": int(radius),
            "n_bits": int(n_bits),
            "cutoff": float(cutoff),
            "test_ratio": float(test_ratio),
            "seed": int(seed),
        },
    }
    with open(output_dir / "butina_split_summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    return output_dir


def _load_or_compute_morgan_fingerprints(
    df_filtered,
    ionizable_smiles_column=IONIZABLE_SMILES_COLUMN,
    radius=2,
    n_bits=2048,
    morgan_fps_path=None,
    data_csv=None,
    verbose=True,
):
    """
    Load precomputed Morgan fingerprints and Tanimoto distance matrix from local .pkl if available.
    If the file does not exist or parameters mismatch, compute via modern MorganGenerator and persist locally.
    """
    from pathlib import Path
    import pickle
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    target_path = None
    if morgan_fps_path and str(morgan_fps_path).strip():
        target_path = Path(morgan_fps_path)
    elif data_csv:
        csv_p = Path(data_csv)
        candidates = [
            csv_p.parent / f"{csv_p.stem}_morgan_fingerprints_r{radius}_b{n_bits}.pkl",
            csv_p.parent / f"{csv_p.stem}_morgan_fingerprints.pkl",
        ]
        if "lnp-447" in csv_p.stem.lower():
            candidates.extend([
                Path("data") / f"LNP-447_morgan_fingerprints_r{radius}_b{n_bits}.pkl",
                Path("data") / "LNP-447_morgan_fingerprints.pkl",
            ])
        for cand in candidates:
            if cand.exists():
                target_path = cand
                break

    # 1. Attempt to load from local precomputed file
    if target_path and target_path.exists():
        try:
            with open(target_path, "rb") as f:
                loaded = pickle.load(f)
            if (
                isinstance(loaded, dict)
                and loaded.get("radius") == radius
                and loaded.get("n_bits") == n_bits
                and "fingerprints" in loaded
                and len(loaded.get("canonical_all", [])) == len(df_filtered)
            ):
                if verbose:
                    print(f"[Butina] [OK] Successfully loaded precomputed Morgan fingerprint file: {target_path}")
                canonical_smiles = loaded["canonical_all"]
                unique_smiles = loaded["unique_smiles"]
                fingerprints = loaded["fingerprints"]
                distance_matrix = loaded.get("distance_matrix")
                fingerprint_matrix = loaded.get("fingerprint_matrix")

                if distance_matrix is None:
                    distance_matrix = []
                    for idx in range(1, len(fingerprints)):
                        similarities = DataStructs.BulkTanimotoSimilarity(
                            fingerprints[idx],
                            fingerprints[:idx],
                        )
                        distance_matrix.extend([1.0 - sim for sim in similarities])

                if fingerprint_matrix is None:
                    fingerprint_matrix = _fingerprints_to_numpy(fingerprints, n_bits)

                return (
                    canonical_smiles,
                    unique_smiles,
                    fingerprints,
                    distance_matrix,
                    fingerprint_matrix,
                )
        except Exception as exc:
            warnings.warn(f"Failed to read precomputed fingerprint file, falling back to real-time computation: {exc}")

    # 2. Compute Morgan fingerprints from SMILES in real-time (using modern rdFingerprintGenerator)
    if verbose:
        print("[Butina] [INFO] Precomputed fingerprint file not found or mismatched, generating Morgan fingerprints in real-time...")

    smiles_series = (
        df_filtered[ionizable_smiles_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": ""})
    )
    canonical_smiles = []
    for raw_smiles in smiles_series:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            canonical_smiles.append(None)
        else:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))

    unique_smiles = [s for s in dict.fromkeys(canonical_smiles) if s is not None]

    try:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        get_fp = mfpgen.GetFingerprint
    except Exception:
        from rdkit.Chem import AllChem
        get_fp = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=n_bits)

    fingerprints = []
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(get_fp(mol))

    distance_matrix = []
    for idx in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[idx],
            fingerprints[:idx],
        )
        distance_matrix.extend([1.0 - sim for sim in similarities])

    fingerprint_matrix = _fingerprints_to_numpy(fingerprints, n_bits)

    auto_save_path = target_path or (
        Path(data_csv).parent / f"{Path(data_csv).stem}_morgan_fingerprints.pkl"
        if data_csv
        else None
    )
    if auto_save_path:
        try:
            auto_save_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "canonical_all": canonical_smiles,
                "unique_smiles": unique_smiles,
                "fingerprints": fingerprints,
                "distance_matrix": distance_matrix,
                "fingerprint_matrix": fingerprint_matrix,
                "radius": radius,
                "n_bits": n_bits,
                "dataset": Path(data_csv).stem if data_csv else "dataset",
            }
            with open(auto_save_path, "wb") as f:
                pickle.dump(payload, f)
            if verbose:
                print(f"[Butina] [SAVE] Morgan fingerprints saved to: {auto_save_path}")
        except Exception as exc:
            warnings.warn(f"Failed to automatically save precomputed fingerprint file: {exc}")

    return (
        canonical_smiles,
        unique_smiles,
        fingerprints,
        distance_matrix,
        fingerprint_matrix,
    )


def get_butina_split_data(
    data_csv=DEFAULT_DATA_CSV,
    embedding_npy=DEFAULT_EMBEDDING_NPY,
    target_column=TARGET_COLUMN,
    target_classes=TARGET_CLASSES,
    ionizable_smiles_column=IONIZABLE_SMILES_COLUMN,
    radius=2,
    n_bits=2048,
    cutoff=0.6,
    val_ratio=0.0,
    test_ratio=0.1,
    min_train_ratio=0.75,
    seed=42,
    visualize=False,
    figure_path=None,
    split_output_dir=None,
    projection_metric="jaccard",
    morgan_fps_path=None,
    verbose=True,
):
    """
    OOD split based on Butina clustering of ionizable lipids (pure single-cluster independent knapsack mode).

    Parameters
    ----------
    data_csv : str or Path, default=DEFAULT_DATA_CSV
        Path to LNP label CSV.
    embedding_npy : str or Path, default=DEFAULT_EMBEDDING_NPY
        Path to LNP embedding .npy file.
    target_column : str, default=TARGET_COLUMN
        Target column name in CSV.
    target_classes : tuple, default=TARGET_CLASSES
        Target class labels.
    ionizable_smiles_column : str, default=IONIZABLE_SMILES_COLUMN
        Column name for ionizable lipid SMILES.
    radius : int, default=2
        Morgan fingerprint radius.
    n_bits : int, default=2048
        Morgan fingerprint bit length.
    cutoff : float, default=0.6
        Butina clustering distance cutoff.
    val_ratio : float, default=0.0
        Target validation set fraction (0.0 means no validation set).
    test_ratio : float, default=0.1
        Target test set fraction.
    min_train_ratio : float, default=0.75
        Minimum fraction of samples to retain in the training set.
    seed : int, default=42
        Random seed.
    visualize : bool, default=False
        Whether to generate and save 2D projection plot.
    figure_path : str or Path, optional
        Path to save 2D projection plot.
    split_output_dir : str or Path, optional
        Directory to save split metadata and sample IDs.
    projection_metric : str, default="jaccard"
        Distance metric for 2D projection ("jaccard" or "euclidean").
    morgan_fps_path : str or Path, optional
        Path to precomputed Morgan fingerprint .pkl file. If None, auto-detected.
    verbose : bool, default=True
        Whether to print logging information.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Running Butina split requires RDKit. "
            "Install via conda: conda install -c conda-forge rdkit"
        ) from exc

    df_filtered, x_filtered, y_filtered = _load_aligned_lnp_dataset(
        data_csv=data_csv,
        embedding_npy=embedding_npy,
        target_column=target_column,
        target_classes=target_classes,
    )

    (
        canonical_smiles,
        unique_smiles,
        fingerprints,
        distance_matrix,
        fingerprint_matrix,
    ) = _load_or_compute_morgan_fingerprints(
        df_filtered=df_filtered,
        ionizable_smiles_column=ionizable_smiles_column,
        radius=radius,
        n_bits=n_bits,
        morgan_fps_path=morgan_fps_path,
        data_csv=data_csv,
        verbose=verbose,
    )

    df_filtered = df_filtered.copy()
    df_filtered["_ionizable_canonical_smiles"] = canonical_smiles

    valid_smiles_mask = df_filtered["_ionizable_canonical_smiles"].notna()
    fixed_train_count = int((~valid_smiles_mask).sum())

    if valid_smiles_mask.sum() == 0:
        raise ValueError("[ERROR] No valid ionizable lipid SMILES available for Butina clustering.")

    clusters = Butina.ClusterData(
        distance_matrix,
        len(fingerprints),
        cutoff,
        isDistData=True,
    )
    if len(clusters) == 0:
        raise ValueError("[ERROR] Butina clustering produced no clusters; please check inputs or adjust cutoff.")

    smiles_to_cluster = {}
    for cluster_id, member_indices in enumerate(clusters):
        for member_idx in member_indices:
            smiles_to_cluster[unique_smiles[member_idx]] = cluster_id

    df_filtered["_cluster_id"] = df_filtered["_ionizable_canonical_smiles"].map(smiles_to_cluster)

    cluster_sizes = (
        df_filtered.loc[valid_smiles_mask]
        .groupby("_cluster_id")
        .size()
        .astype(int)
        .to_dict()
    )

    target_test_size = max(1, int(round(len(df_filtered) * float(test_ratio))))
    target_val_size = (
        max(1, int(round(len(df_filtered) * float(val_ratio))))
        if (val_ratio and float(val_ratio) > 0.0)
        else 0
    )

    min_train_count = (
        int(round(len(df_filtered) * float(min_train_ratio)))
        if (min_train_ratio and float(min_train_ratio) > 0.0)
        else 0
    )
    min_train_count_for_test = max(0, min_train_count + target_val_size)

    test_cluster_ids = _subset_sum_cluster_split(
        cluster_sizes,
        target_size=target_test_size,
        seed=seed,
        fixed_train_count=fixed_train_count,
        min_train_count=min_train_count_for_test,
    )

    val_cluster_ids = set()
    if target_val_size > 0:
        remaining_cluster_sizes = {
            cid: size for cid, size in cluster_sizes.items() if cid not in test_cluster_ids
        }
        if remaining_cluster_sizes:
            val_cluster_ids = _subset_sum_cluster_split(
                remaining_cluster_sizes,
                target_size=target_val_size,
                seed=seed + 1,
                fixed_train_count=fixed_train_count,
                min_train_count=min_train_count,
            )

    df_filtered["_split"] = "train"
    df_filtered.loc[df_filtered["_cluster_id"].isin(val_cluster_ids), "_split"] = "val"
    df_filtered.loc[df_filtered["_cluster_id"].isin(test_cluster_ids), "_split"] = "test"
    df_filtered.loc[~valid_smiles_mask, "_split"] = "train"

    # Prevent data leakage: identical canonical SMILES must belong to a single unique split.
    split_per_smiles = (
        df_filtered.loc[valid_smiles_mask, ["_ionizable_canonical_smiles", "_split"]]
        .drop_duplicates()
        .groupby("_ionizable_canonical_smiles")["_split"]
        .nunique()
    )
    if int(split_per_smiles.max()) > 1:
        raise RuntimeError("[ERROR] Detected identical ionizable lipid SMILES across different splits; partitioning failed.")

    train_mask = df_filtered["_split"].eq("train").to_numpy()
    val_mask = df_filtered["_split"].eq("val").to_numpy()
    test_mask = df_filtered["_split"].eq("test").to_numpy()

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "[ERROR] Training or test set is empty after Butina split. "
            "Suggest adjusting cutoff or checking valid SMILES count."
        )
    if val_ratio and float(val_ratio) > 0.0 and val_mask.sum() == 0:
        raise ValueError(
            "[ERROR] Validation set is empty after Butina split. "
            "Suggest adjusting cutoff or val_ratio."
        )

    x_train = x_filtered[train_mask]
    x_val = x_filtered[val_mask] if val_mask.any() else None
    x_test = x_filtered[test_mask]
    y_train = y_filtered[train_mask]
    y_val = y_filtered[val_mask] if val_mask.any() else None
    y_test = y_filtered[test_mask]

    if verbose:
        print("=" * 80)
        print("[Butina] [OK] OOD split completed")
        print(f"[Butina] Aligned sample count: {len(df_filtered)}")
        print(f"[Butina] Unique canonical ionizable lipids: {len(unique_smiles)}")
        print(f"[Butina] Number of clusters: {len(clusters)}")
        print(f"[Butina] Invalid / unparseable SMILES count: {fixed_train_count}")
        if val_ratio and float(val_ratio) > 0.0:
            print(f"[Butina] Train set: {x_train.shape}, Val set: {x_val.shape}, Test set: {x_test.shape}")
            print(f"[Butina] Actual val ratio: {len(x_val) / len(df_filtered):.4f}, Actual test ratio: {len(x_test) / len(df_filtered):.4f}")
        else:
            print(f"[Butina] Train set: {x_train.shape}, Test set: {x_test.shape}")
            print(f"[Butina] Actual test ratio: {len(x_test) / len(df_filtered):.4f}")
        if fixed_train_count > 0:
            print("[Butina] [INFO] Note: Invalid SMILES were all assigned to the train set to prevent leakage.")

    if visualize:
        figure_path = figure_path or Path(__file__).with_name("butina_split_projection.png")
        unique_cluster_ids = np.array([smiles_to_cluster[smiles] for smiles in unique_smiles])
        conditions = [
            np.isin(unique_cluster_ids, list(test_cluster_ids)),
            np.isin(unique_cluster_ids, list(val_cluster_ids)),
        ]
        choices = ["test", "val"]
        unique_splits = np.select(conditions, choices, default="train")
        fingerprint_matrix = _fingerprints_to_numpy(fingerprints, n_bits=n_bits)
        saved_path = _save_butina_projection(
            fingerprint_matrix=fingerprint_matrix,
            unique_splits=unique_splits,
            figure_path=figure_path,
            seed=seed,
            cutoff=cutoff,
            projection_metric=projection_metric,
        )
        if verbose:
            print(f"[Butina] [SAVE] Visualization plot saved to: {saved_path}")

    if split_output_dir is not None:
        saved_dir = _save_split_metadata(
            df_filtered=df_filtered,
            target_column=target_column,
            ionizable_smiles_column=ionizable_smiles_column,
            output_dir=split_output_dir,
            radius=radius,
            n_bits=n_bits,
            cutoff=cutoff,
            test_ratio=test_ratio,
            seed=seed,
        )
        if verbose:
            print(f"[Butina] [SAVE] Split sample metadata saved to: {saved_dir}")

    if val_ratio and float(val_ratio) > 0.0:
        return x_train, x_val, x_test, y_train, y_val, y_test
    return x_train, x_test, y_train, y_test


def parse_args():
    parser = argparse.ArgumentParser(description="Tool for Butina clustering-based LNP OOD split")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV), help=f"Path to LNP label CSV (default: {DEFAULT_DATA_CSV})")
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
        help=f"Path to LNP embedding .npy file (default: {DEFAULT_EMBEDDING_NPY})",
    )
    parser.add_argument("--cutoff", type=float, default=0.6, help="Butina distance cutoff (default: 0.6)")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius (default: 2)")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint bit length (default: 2048)")
    parser.add_argument("--val_ratio", type=float, default=0.0, help="Target validation set ratio, 0.0 means no validation set (default: 0.0)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Target test set ratio (default: 0.1)")
    parser.add_argument("--min_train_ratio", type=float, default=0.75, help="Minimum ratio of training samples to retain (default: 0.75)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--figure_path",
        type=str,
        default=str(Path(__file__).with_name("butina_split_projection.png")),
        help="Path to save 2D projection plot (default: butina_split_projection.png)",
    )
    parser.add_argument(
        "--projection_metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "euclidean"],
        help="Distance metric for 2D projection, 'jaccard' corresponds to molecular Tanimoto distance (default: jaccard)",
    )
    parser.add_argument(
        "--split_output_dir",
        type=str,
        default=str(Path(__file__).with_name("butina_split_outputs")),
        help="Output directory for split sample IDs and metadata (default: butina_split_outputs)",
    )
    parser.add_argument(
        "--morgan_fps_path",
        type=str,
        default="",
        help="Path to precomputed Morgan fingerprint .pkl file (leave empty to auto-detect *_morgan_fingerprints.pkl in dataset directory)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ret = get_butina_split_data(
        data_csv=args.data_csv,
        embedding_npy=args.embedding_npy,
        radius=args.radius,
        n_bits=args.n_bits,
        cutoff=args.cutoff,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_train_ratio=args.min_train_ratio,
        seed=args.seed,
        morgan_fps_path=args.morgan_fps_path,
        visualize=True,
        figure_path=args.figure_path,
        split_output_dir=args.split_output_dir,
        projection_metric=args.projection_metric,
        verbose=True,
    )

    if args.val_ratio > 0.0:
        x_train, x_val, x_test, y_train, y_val, y_test = ret
    else:
        x_train, x_test, y_train, y_test = ret
        x_val, y_val = None, None

    print("=" * 80)
    print("[Butina] Standalone Run Summary")
    print(f"[Butina] X_train shape: {x_train.shape}")
    if x_val is not None:
        print(f"[Butina] X_val shape: {x_val.shape}")
    print(f"[Butina] X_test shape: {x_test.shape}")
    print(f"[Butina] y_train distribution:\n{pd.Series(y_train).value_counts().to_string()}")
    if y_val is not None:
        print(f"[Butina] y_val distribution:\n{pd.Series(y_val).value_counts().to_string()}")
    print(f"[Butina] y_test distribution:\n{pd.Series(y_test).value_counts().to_string()}")
