import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_CSV = Path("data/LNP-447.csv")
DEFAULT_EMBEDDING_NPY = Path("data/LNP-447-ft-final.npy")

TARGET_COLUMN = "Organ"
TARGET_CLASSES = ("Liver", "Lung", "Spleen", "None")


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

    Maintains alignment rules to prevent sample misalignment across training routines.
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
    Approximate target test set size as closely as possible without splitting clusters.

    fixed_train_count indicates samples already assigned to train (e.g. HDBSCAN noise points).
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
    """Prefer UMAP for 2D visualization projection, falling back to PCA."""
    features = np.asarray(features, dtype=np.float32)

    if prefer_umap:
        try:
            import umap

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(20, max(2, len(features) - 1)),
                min_dist=0.15,
                metric="euclidean",
                random_state=seed,
            )
            return reducer.fit_transform(features), "UMAP"
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"UMAP unavailable, falling back to PCA. Reason: {exc}")

    reducer = PCA(n_components=2, random_state=seed)
    return reducer.fit_transform(features), "PCA"


def _build_hdbscan_clusterer(
    min_cluster_size,
    min_samples,
    metric,
    cluster_selection_epsilon,
):
    """
    Prefer external hdbscan package; fall back to sklearn.cluster.HDBSCAN if not installed.
    """
    try:
        import hdbscan

        return hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            prediction_data=False,
        ), "hdbscan"
    except ImportError:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        return SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            allow_single_cluster=False,
            n_jobs=-1,
        ), "sklearn.cluster.HDBSCAN"


def _save_hdbscan_projection(split_labels, features_for_plot, figure_path, seed, backend_name):
    import matplotlib.pyplot as plt

    projection, method_name = _project_to_2d(
        features_for_plot,
        seed=seed,
        prefer_umap=True,
    )

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    style_map = {
        "train": {"color": "#1f77b4", "marker": "o", "label": "Train"},
        "test": {"color": "#d62728", "marker": "^", "label": "Test"},
        "noise": {"color": "#7f7f7f", "marker": "x", "label": "Noise"},
    }

    for split_name, style in style_map.items():
        mask = split_labels == split_name
        if not np.any(mask):
            continue

        plt.scatter(
            projection[mask, 0],
            projection[mask, 1],
            c=style["color"],
            marker=style["marker"],
            s=52 if split_name != "noise" else 42,
            alpha=0.82,
            label=f"{style['label']} ({int(mask.sum())})",
        )

    plt.title(f"HDBSCAN OOD Split Projection ({method_name}, {backend_name})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path


def get_hdbscan_split_data(
    data_csv=DEFAULT_DATA_CSV,
    embedding_npy=DEFAULT_EMBEDDING_NPY,
    target_column=TARGET_COLUMN,
    target_classes=TARGET_CLASSES,
    test_ratio=0.1,
    seed=42,
    min_cluster_size=12,
    min_samples=None,
    metric="euclidean",
    cluster_selection_epsilon=0.0,
    visualize=False,
    figure_path=None,
    verbose=True,
):
    """
    Perform OOD dataset split based on HDBSCAN clustering over overall LNP embeddings.

    Returns
    -------
    X_train, X_test, y_train, y_test

    Key Anti-leakage Logic
    ----------------------
    - Cluster in full LNP embedding space first, then split train/test cluster by cluster.
    - Any single cluster will be wholly placed into either train or test, never split across.
    - HDBSCAN noise points (-1) are assigned to training set by default to avoid placing outlier points into test set.
    """
    _, x_filtered, y_filtered = _load_aligned_lnp_dataset(
        data_csv=data_csv,
        embedding_npy=embedding_npy,
        target_column=target_column,
        target_classes=target_classes,
    )

    scaler = StandardScaler()
    x_cluster = scaler.fit_transform(x_filtered)

    clusterer, backend_name = _build_hdbscan_clusterer(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
    )
    cluster_labels = clusterer.fit_predict(x_cluster)

    noise_mask = cluster_labels == -1
    non_noise_mask = ~noise_mask
    fixed_train_count = int(noise_mask.sum())

    group_sizes = pd.Series(cluster_labels[non_noise_mask]).value_counts().astype(int).to_dict()
    target_test_size = max(1, int(round(len(x_filtered) * float(test_ratio))))
    test_cluster_ids = _subset_sum_cluster_split(
        group_sizes,
        target_size=target_test_size,
        seed=seed,
        fixed_train_count=fixed_train_count,
    )

    split_labels = np.full(len(cluster_labels), "train", dtype=object)
    split_labels[noise_mask] = "noise"
    split_labels[np.isin(cluster_labels, list(test_cluster_ids))] = "test"
    split_labels[noise_mask] = "noise"

    train_mask = split_labels != "test"
    test_mask = split_labels == "test"

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(
            "Train or test set is empty after HDBSCAN split. "
            "Suggest adjusting min_cluster_size / min_samples / test_ratio."
        )

    x_train = x_filtered[train_mask]
    x_test = x_filtered[test_mask]
    y_train = y_filtered[train_mask]
    y_test = y_filtered[test_mask]

    if verbose:
        cluster_count = len(set(cluster_labels.tolist()) - {-1})
        print("=" * 80)
        print("[HDBSCAN] OOD split completed")
        print(f"[HDBSCAN] Sample count: {len(x_filtered)}")
        print(f"[HDBSCAN] Non-noise clusters: {cluster_count}")
        print(f"[HDBSCAN] Noise points count: {fixed_train_count}")
        print(f"[HDBSCAN] Backend implementation: {backend_name}")
        print(f"[HDBSCAN] Train set shape: {x_train.shape}, Test set shape: {x_test.shape}")
        print(f"[HDBSCAN] Actual test ratio: {len(x_test) / len(x_filtered):.4f}")

    if visualize:
        figure_path = figure_path or Path(__file__).with_name("hdbscan_split_projection.png")
        saved_path = _save_hdbscan_projection(
            split_labels=split_labels,
            features_for_plot=x_cluster,
            figure_path=figure_path,
            seed=seed,
            backend_name=backend_name,
        )
        if verbose:
            print(f"[HDBSCAN] Visualization saved: {saved_path}")

    return x_train, x_test, y_train, y_test


def parse_args():
    parser = argparse.ArgumentParser(description="LNP OOD splitting tool based on HDBSCAN clustering")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV), help="Path to LNP label CSV file")
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
        help="Path to LNP embedding .npy file",
    )
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Target test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min_cluster_size", type=int, default=12, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min_samples", type=int, default=None, help="HDBSCAN min_samples")
    parser.add_argument("--metric", type=str, default="euclidean", help="HDBSCAN distance metric")
    parser.add_argument(
        "--cluster_selection_epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon",
    )
    parser.add_argument(
        "--figure_path",
        type=str,
        default=str(Path(__file__).with_name("hdbscan_split_projection.png")),
        help="Save path for 2D projection figure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x_train, x_test, y_train, y_test = get_hdbscan_split_data(
        data_csv=args.data_csv,
        embedding_npy=args.embedding_npy,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        visualize=True,
        figure_path=args.figure_path,
        verbose=True,
    )

    print("=" * 80)
    print("[HDBSCAN] Standalone execution summary")
    print(f"[HDBSCAN] X_train shape: {x_train.shape}")
    print(f"[HDBSCAN] X_test shape: {x_test.shape}")
    print(f"[HDBSCAN] y_train distribution:\n{pd.Series(y_train).value_counts().to_string()}")
    print(f"[HDBSCAN] y_test distribution:\n{pd.Series(y_test).value_counts().to_string()}")
