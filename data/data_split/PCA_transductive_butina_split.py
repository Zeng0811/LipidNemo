import argparse
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
        raise FileNotFoundError(f"File does not exist: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".npy":
        return np.load(file_path)
    if ext == ".npz":
        with np.load(file_path) as loaded:
            keys = list(loaded.keys())
            return loaded["arr_0"] if "arr_0" in keys else loaded[keys[0]]
    if ext == ".csv":
        return pd.read_csv(file_path).values.astype(np.float32)

    raise ValueError(f"Unsupported file format: {file_path.suffix}")


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

    Preserves core behavior:
    1. Truncate alignment by minimum length if row counts mismatch;
    2. Retain target classes for training and evaluation.
    """
    data_csv = Path(data_csv)
    embedding_npy = Path(embedding_npy)

    df = pd.read_csv(data_csv)
    x_all = data_load(embedding_npy)

    if x_all.ndim != 2:
        raise ValueError(f"Embedding must be a 2D array, current shape={x_all.shape}")

    aligned_rows = min(len(df), len(x_all))
    if len(df) != len(x_all):
        warnings.warn(
            "Row count mismatch between CSV and embedding. Truncated to minimum aligned length."
            f" CSV={len(df)}, embedding={len(x_all)}, aligned={aligned_rows}"
        )

    df = df.iloc[:aligned_rows].copy()
    x_all = x_all[:aligned_rows]

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
        raise ValueError("No available samples after filtering target labels.")

    return df_filtered, x_filtered, y_filtered


def _subset_sum_cluster_split(group_sizes, target_size, seed=42, fixed_train_count=0):
    """
    Approximate target test set sample count using subset-sum DP without splitting clusters.

    Parameters
    ----------
    group_sizes:
        {group_id: sample_count}
    target_size:
        Target test set sample count
    fixed_train_count:
        Number of samples forcibly assigned to training set (e.g. invalid SMILES).
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


def _project_to_2d(features, seed=42, prefer_umap=True):
    """Prefer UMAP for 2D projection, falling back to PCA if UMAP is unavailable."""
    features = np.asarray(features, dtype=np.float32)

    if prefer_umap:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, len(features) - 1)),
                min_dist=0.1,
                metric="euclidean",
                random_state=seed,
            )
            return reducer.fit_transform(features), "UMAP"
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"UMAP unavailable, falling back to PCA. Reason: {exc}")

    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(features), "PCA"


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
):
    import matplotlib.pyplot as plt

    projection, method_name = _project_to_2d(
        fingerprint_matrix,
        seed=seed,
        prefer_umap=True,
    )

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    style_map = {
        "train": {"color": "#1f77b4", "marker": "o", "label": "Train"},
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


def get_butina_split_data(
    data_csv=DEFAULT_DATA_CSV,
    embedding_npy=DEFAULT_EMBEDDING_NPY,
    target_column=TARGET_COLUMN,
    target_classes=TARGET_CLASSES,
    ionizable_smiles_column=IONIZABLE_SMILES_COLUMN,
    radius=2,
    n_bits=2048,
    cutoff=0.6,
    test_ratio=0.1,
    seed=42,
    visualize=False,
    figure_path=None,
    verbose=True,
):
    """
    Perform OOD dataset split based on ionizable lipid Butina clustering.

    Returns
    -------
    X_train, X_test, y_train, y_test

    Key Anti-leakage Logic
    ----------------------
    - Canonicalize ionizable lipid SMILES first to avoid splitting identical structures across sets.
    - Cluster at the 'unique ionizable lipid' level first, then map whole clusters back to original LNP formulations.
    - Identical canonical SMILES will never appear in both train and test sets.
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Running Butina split requires RDKit installed. "
            "Install via: conda install -c conda-forge rdkit"
        ) from exc

    df_filtered, x_filtered, y_filtered = _load_aligned_lnp_dataset(
        data_csv=data_csv,
        embedding_npy=embedding_npy,
        target_column=target_column,
        target_classes=target_classes,
    )

    smiles_series = (
        df_filtered[ionizable_smiles_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": ""})
    )

    canonical_smiles = []
    invalid_smiles = []
    for raw_smiles in smiles_series:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            canonical_smiles.append(None)
            invalid_smiles.append(raw_smiles)
        else:
            canonical_smiles.append(Chem.MolToSmiles(mol, canonical=True))

    df_filtered = df_filtered.copy()
    df_filtered["_ionizable_canonical_smiles"] = canonical_smiles

    valid_smiles_mask = df_filtered["_ionizable_canonical_smiles"].notna()
    fixed_train_count = int((~valid_smiles_mask).sum())

    if valid_smiles_mask.sum() == 0:
        raise ValueError("No valid ionizable lipid SMILES available for Butina clustering.")

    unique_smiles = (
        df_filtered.loc[valid_smiles_mask, "_ionizable_canonical_smiles"]
        .drop_duplicates()
        .tolist()
    )

    fingerprints = []
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        fingerprints.append(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        )

    distance_matrix = []
    for idx in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[idx],
            fingerprints[:idx],
        )
        distance_matrix.extend([1.0 - sim for sim in similarities])

    clusters = Butina.ClusterData(
        distance_matrix,
        len(fingerprints),
        cutoff,
        isDistData=True,
    )
    if len(clusters) == 0:
        raise ValueError("Butina produced no clusters. Please check inputs or adjust cutoff.")

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
    test_cluster_ids = _subset_sum_cluster_split(
        cluster_sizes,
        target_size=target_test_size,
        seed=seed,
        fixed_train_count=fixed_train_count,
    )

    df_filtered["_split"] = "train"
    df_filtered.loc[df_filtered["_cluster_id"].isin(test_cluster_ids), "_split"] = "test"
    df_filtered.loc[~valid_smiles_mask, "_split"] = "train"

    # Prevent data leakage: identical canonical SMILES can only belong to a single split.
    split_per_smiles = (
        df_filtered.loc[valid_smiles_mask, ["_ionizable_canonical_smiles", "_split"]]
        .drop_duplicates()
        .groupby("_ionizable_canonical_smiles")["_split"]
        .nunique()
    )
    if int(split_per_smiles.max()) > 1:
        raise RuntimeError("Detected identical ionizable lipid SMILES spanning train/test sets. Splitting failed.")

    train_mask = df_filtered["_split"].eq("train").to_numpy()
    test_mask = df_filtered["_split"].eq("test").to_numpy()

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "Train or test set is empty after Butina split. "
            "Suggest adjusting cutoff or checking valid SMILES count."
        )

    x_train = x_filtered[train_mask]
    x_test = x_filtered[test_mask]
    y_train = y_filtered[train_mask]
    y_test = y_filtered[test_mask]

    if verbose:
        print("=" * 80)
        print("[Butina] OOD split completed")
        print(f"[Butina] Aligned sample count: {len(df_filtered)}")
        print(f"[Butina] Unique canonical ionizable lipids: {len(unique_smiles)}")
        print(f"[Butina] Number of clusters: {len(clusters)}")
        print(f"[Butina] Invalid/unparseable SMILES count: {fixed_train_count}")
        print(f"[Butina] Train set shape: {x_train.shape}, Test set shape: {x_test.shape}")
        print(f"[Butina] Actual test ratio: {len(x_test) / len(df_filtered):.4f}")
        if invalid_smiles:
            print("[Butina] Note: Invalid SMILES are fixed in the training set to prevent leakage.")

    if visualize:
        figure_path = figure_path or Path(__file__).with_name("butina_split_projection.png")
        unique_cluster_ids = np.array([smiles_to_cluster[smiles] for smiles in unique_smiles])
        unique_splits = np.where(np.isin(unique_cluster_ids, list(test_cluster_ids)), "test", "train")
        fingerprint_matrix = _fingerprints_to_numpy(fingerprints, n_bits=n_bits)
        saved_path = _save_butina_projection(
            fingerprint_matrix=fingerprint_matrix,
            unique_splits=unique_splits,
            figure_path=figure_path,
            seed=seed,
            cutoff=cutoff,
        )
        if verbose:
            print(f"[Butina] Visualization saved: {saved_path}")

    return x_train, x_test, y_train, y_test


def parse_args():
    parser = argparse.ArgumentParser(description="LNP OOD splitting tool based on Butina clustering")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV), help="Path to LNP label CSV file")
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
        help="Path to LNP embedding .npy file",
    )
    parser.add_argument("--cutoff", type=float, default=0.6, help="Butina distance cutoff")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan fingerprint bit size")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Target test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--figure_path",
        type=str,
        default=str(Path(__file__).with_name("butina_split_projection.png")),
        help="Save path for 2D projection figure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x_train, x_test, y_train, y_test = get_butina_split_data(
        data_csv=args.data_csv,
        embedding_npy=args.embedding_npy,
        radius=args.radius,
        n_bits=args.n_bits,
        cutoff=args.cutoff,
        test_ratio=args.test_ratio,
        seed=args.seed,
        visualize=True,
        figure_path=args.figure_path,
        verbose=True,
    )

    print("=" * 80)
    print("[Butina] Standalone execution summary")
    print(f"[Butina] X_train shape: {x_train.shape}")
    print(f"[Butina] X_test shape: {x_test.shape}")
    print(f"[Butina] y_train distribution:\n{pd.Series(y_train).value_counts().to_string()}")
    print(f"[Butina] y_test distribution:\n{pd.Series(y_test).value_counts().to_string()}")
