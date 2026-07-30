import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_CSV = Path("data/LNP-447.csv")
DEFAULT_EMBEDDING_NPY = Path("data/LNP-447-ft-final.npy")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("split_outputs") / "hdbscan_hpo"

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

    Hyperparameter optimization repeatedly reuses the same dataset split, so we fix:
    - Sample alignment
    - Label filtering
    - Feature array truncation
    All subsequent index outputs are based on this aligned, filtered dataset.
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
    df["_aligned_index"] = np.arange(aligned_rows)
    x_all = x_all[:aligned_rows]

    df[target_column] = _clean_target_series(df[target_column])
    mask = df[target_column].isin(target_classes)

    df_filtered = df.loc[mask].copy().reset_index(drop=True)
    df_filtered["_filtered_index"] = np.arange(len(df_filtered))
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
    """Approximate target test set size as closely as possible without splitting clusters."""
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


def _save_projection_points(plot_df, coordinate_path, projection_method, target_column):
    coordinate_path = Path(coordinate_path)
    coordinate_path.parent.mkdir(parents=True, exist_ok=True)

    coordinate_df = plot_df.copy()
    coordinate_df["projection_method"] = projection_method
    coordinate_df = coordinate_df[
        [
            "filtered_index",
            "aligned_index",
            target_column,
            "split",
            "cluster_id",
            "is_test_cluster",
            "is_noise",
            "x",
            "y",
            "projection_method",
        ]
    ].copy()
    coordinate_df.to_csv(coordinate_path, index=False, encoding="utf-8-sig")
    return coordinate_path


def _save_split_visualization(
    split_table,
    cluster_summary,
    features_for_plot,
    figure_path,
    seed,
    backend_name,
    target_column,
    coordinate_path=None,
):
    import matplotlib.pyplot as plt

    projection, method_name = _project_to_2d(features_for_plot, seed=seed, prefer_umap=True)
    plot_df = split_table.copy()
    plot_df["x"] = projection[:, 0]
    plot_df["y"] = projection[:, 1]
    if coordinate_path is not None:
        _save_projection_points(plot_df, coordinate_path, method_name, target_column)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        gridspec_kw={"width_ratios": [2.5, 1.1]},
    )

    scatter_ax, bar_ax = axes
    style_map = {
        "train": {"color": "#1f77b4", "marker": "o", "label": "Train"},
        "test": {"color": "#d62728", "marker": "^", "label": "Test"},
        "noise": {"color": "#7f7f7f", "marker": "x", "label": "Noise"},
    }

    for split_name, style in style_map.items():
        mask = plot_df["split"].eq(split_name)
        if not mask.any():
            continue
        edgecolors = "black" if split_name == "test" else style["color"]
        scatter_ax.scatter(
            plot_df.loc[mask, "x"],
            plot_df.loc[mask, "y"],
            c=style["color"],
            marker=style["marker"],
            s=52 if split_name != "noise" else 40,
            alpha=0.82,
            edgecolors=edgecolors,
            linewidths=0.4,
            label=f"{style['label']} ({int(mask.sum())})",
        )

    centroid_df = (
        plot_df.loc[plot_df["cluster_id"] >= 0]
        .groupby("cluster_id")[["x", "y"]]
        .mean()
        .reset_index()
    )
    test_cluster_ids = set(
        cluster_summary.loc[cluster_summary["split"].eq("test"), "cluster_id"]
        .astype(int)
        .tolist()
    )
    for _, row in centroid_df.iterrows():
        cluster_id = int(row["cluster_id"])
        text_color = "#d62728" if cluster_id in test_cluster_ids else "#2f2f2f"
        scatter_ax.text(
            row["x"],
            row["y"],
            f"C{cluster_id}",
            fontsize=8.5,
            color=text_color,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": text_color, "alpha": 0.78},
        )

    scatter_ax.set_title(
        f"HDBSCAN 2D Projection ({method_name}, {backend_name})\n"
        f"Total clusters={len(centroid_df)}, Test clusters={sorted(test_cluster_ids)}"
    )
    scatter_ax.set_xlabel("Component 1")
    scatter_ax.set_ylabel("Component 2")
    scatter_ax.legend(loc="best")

    bar_df = cluster_summary.copy()
    bar_df["cluster_label"] = bar_df["cluster_id"].apply(
        lambda value: "Noise" if int(value) < 0 else f"C{int(value)}"
    )
    bar_df = bar_df.sort_values(by=["cluster_id"]).reset_index(drop=True)
    bar_colors = bar_df["split"].map({"train": "#1f77b4", "test": "#d62728"}).fillna("#7f7f7f")

    bar_ax.barh(bar_df["cluster_label"], bar_df["sample_count"], color=bar_colors)
    for _, row in bar_df.iterrows():
        bar_ax.text(
            row["sample_count"] + 0.3,
            row["cluster_label"],
            str(int(row["sample_count"])),
            va="center",
            fontsize=8.5,
        )

    bar_ax.set_title("Cluster Size Distribution")
    bar_ax.set_xlabel("Sample Count")
    bar_ax.set_ylabel("Cluster ID")

    fig.suptitle("HDBSCAN HPO Split Overview", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return figure_path


def _save_outputs(
    output_dir,
    save_prefix,
    split_table,
    cluster_summary,
    summary,
    figure_path=None,
    coordinate_path=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = output_dir / f"{save_prefix}_sample_split.csv"
    y_train_path = output_dir / f"{save_prefix}_y_train.csv"
    y_test_path = output_dir / f"{save_prefix}_y_test.csv"
    cluster_path = output_dir / f"{save_prefix}_cluster_summary.csv"
    summary_path = output_dir / f"{save_prefix}_summary.json"

    split_table.to_csv(split_path, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(cluster_path, index=False, encoding="utf-8-sig")

    train_table = split_table.loc[split_table["split"].eq("train"), ["filtered_index", "aligned_index", TARGET_COLUMN]]
    test_table = split_table.loc[split_table["split"].eq("test"), ["filtered_index", "aligned_index", TARGET_COLUMN]]
    train_table.to_csv(y_train_path, index=False, encoding="utf-8-sig")
    test_table.to_csv(y_test_path, index=False, encoding="utf-8-sig")

    payload = dict(summary)
    payload["figure_path"] = str(figure_path) if figure_path else None
    payload["projection_points_path"] = str(coordinate_path) if coordinate_path else None
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return {
        "split_path": str(split_path),
        "y_train_path": str(y_train_path),
        "y_test_path": str(y_test_path),
        "cluster_summary_path": str(cluster_path),
        "summary_path": str(summary_path),
        "figure_path": str(figure_path) if figure_path else None,
        "projection_points_path": str(coordinate_path) if coordinate_path else None,
    }


def get_hdbscan_hpo_split(
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
    output_dir=None,
    save_prefix=None,
    verbose=True,
):
    """
    Perform HDBSCAN clustering split for reuse during hyperparameter optimization (HPO).

    Key outputs:
    - Train/test filtered indices and aligned indices
    - y_train / y_test labels
    - Cluster ID, split, and noise flag for each sample
    - Cluster IDs assigned to the test set
    """
    df, x_filtered, y_filtered = _load_aligned_lnp_dataset(
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
    noise_count = int(noise_mask.sum())

    group_sizes = pd.Series(cluster_labels[non_noise_mask]).value_counts().astype(int).to_dict()
    target_test_size = max(1, int(round(len(df) * float(test_ratio))))
    test_cluster_ids = _subset_sum_cluster_split(
        group_sizes,
        target_size=target_test_size,
        seed=seed,
        fixed_train_count=noise_count,
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

    df = df.copy()
    df["cluster_id"] = cluster_labels.astype(int)
    df["split"] = split_labels
    df["is_test_cluster"] = df["cluster_id"].isin(list(test_cluster_ids))
    df["is_noise"] = df["cluster_id"].eq(-1)

    split_table = df[
        [
            "_filtered_index",
            "_aligned_index",
            target_column,
            "split",
            "cluster_id",
            "is_test_cluster",
            "is_noise",
        ]
    ].rename(
        columns={
            "_filtered_index": "filtered_index",
            "_aligned_index": "aligned_index",
        }
    ).copy()

    cluster_summary = (
        split_table.groupby(["cluster_id", "split"], as_index=False)
        .agg(
            sample_count=("filtered_index", "size"),
            y_unique_count=(target_column, "nunique"),
        )
        .sort_values(by=["split", "sample_count", "cluster_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    cluster_summary["is_test_cluster"] = cluster_summary["split"].eq("test")

    figure_path = None
    projection_points_path = None
    if visualize:
        output_base_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        file_stem = save_prefix or f"hdbscan_mcs{min_cluster_size}_seed{seed}"
        figure_path = output_base_dir / f"{file_stem}_visualization.png"
        projection_points_path = output_base_dir / f"{file_stem}_projection_points.csv"
        figure_path = _save_split_visualization(
            split_table=split_table,
            cluster_summary=cluster_summary,
            features_for_plot=x_cluster,
            figure_path=figure_path,
            seed=seed,
            backend_name=backend_name,
            target_column=target_column,
            coordinate_path=projection_points_path,
        )

    non_noise_clusters = sorted(int(x) for x in sorted(set(cluster_labels.tolist()) - {-1}))
    summary = {
        "method": "hdbscan",
        "backend": backend_name,
        "seed": int(seed),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": None if min_samples is None else int(min_samples),
        "metric": metric,
        "cluster_selection_epsilon": float(cluster_selection_epsilon),
        "total_samples": int(len(df)),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "test_ratio_actual": float(test_mask.sum() / len(df)),
        "total_clusters": int(len(non_noise_clusters)),
        "cluster_ids": non_noise_clusters,
        "test_cluster_ids": sorted(int(x) for x in test_cluster_ids),
        "noise_count": int(noise_count),
        "y_train_distribution": pd.Series(y_filtered[train_mask]).value_counts().to_dict(),
        "y_test_distribution": pd.Series(y_filtered[test_mask]).value_counts().to_dict(),
    }

    saved_paths = None
    if output_dir is not None:
        saved_paths = _save_outputs(
            output_dir=output_dir,
            save_prefix=save_prefix or f"hdbscan_mcs{min_cluster_size}_seed{seed}",
            split_table=split_table,
            cluster_summary=cluster_summary,
            summary=summary,
            figure_path=figure_path,
            coordinate_path=projection_points_path,
        )

    if verbose:
        print("=" * 90)
        print("[HDBSCAN-HPO] Dataset split completed")
        print(f"[HDBSCAN-HPO] Total samples: {summary['total_samples']}")
        print(f"[HDBSCAN-HPO] Total clusters: {summary['total_clusters']}")
        print(f"[HDBSCAN-HPO] Test clusters: {summary['test_cluster_ids']}")
        print(f"[HDBSCAN-HPO] Noise points: {summary['noise_count']}")
        print(f"[HDBSCAN-HPO] Train samples: {summary['train_samples']}")
        print(f"[HDBSCAN-HPO] Test samples: {summary['test_samples']}")
        print(f"[HDBSCAN-HPO] y_train distribution: {summary['y_train_distribution']}")
        print(f"[HDBSCAN-HPO] y_test distribution: {summary['y_test_distribution']}")
        if saved_paths:
            print(f"[HDBSCAN-HPO] Output directory: {output_dir}")

    return {
        "train_indices": split_table.loc[split_table["split"].ne("test"), "filtered_index"].to_numpy(),
        "test_indices": split_table.loc[split_table["split"].eq("test"), "filtered_index"].to_numpy(),
        "train_aligned_indices": split_table.loc[split_table["split"].ne("test"), "aligned_index"].to_numpy(),
        "test_aligned_indices": split_table.loc[split_table["split"].eq("test"), "aligned_index"].to_numpy(),
        "y_train": split_table.loc[split_table["split"].ne("test"), target_column].to_numpy(),
        "y_test": split_table.loc[split_table["split"].eq("test"), target_column].to_numpy(),
        "split_table": split_table,
        "cluster_summary": cluster_summary,
        "test_cluster_ids": sorted(int(x) for x in test_cluster_ids),
        "summary": summary,
        "saved_paths": saved_paths,
        "figure_path": str(figure_path) if figure_path else None,
        "projection_points_path": str(projection_points_path) if projection_points_path else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset splitting script based on HDBSCAN clustering for HPO")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV), help="Path to label CSV file")
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
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for split results",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="",
        help="Prefix for output filenames; auto-generated if empty",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Output split results only without saving visualization images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = get_hdbscan_hpo_split(
        data_csv=args.data_csv,
        embedding_npy=args.embedding_npy,
        test_ratio=args.test_ratio,
        seed=args.seed,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric=args.metric,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        visualize=not args.no_visualize,
        output_dir=args.output_dir,
        save_prefix=args.save_prefix or None,
        verbose=True,
    )

    print("=" * 90)
    print("[HDBSCAN-HPO] Standalone execution summary")
    print(f"[HDBSCAN-HPO] y_train first 10 items: {result['y_train'][:10].tolist()}")
    print(f"[HDBSCAN-HPO] y_test first 10 items: {result['y_test'][:10].tolist()}")
