import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


DEFAULT_DATA_CSV = Path(
    r"./data/LNP-447.csv"
)
DEFAULT_EMBEDDING_NPY = Path(
    r"./data/LNP-447-ft-final.npy"
)
DEFAULT_OUTPUT_DIR = Path(
    r"./results/split_outputs/butina_hpo"
)

TARGET_COLUMN = "Organ"
TARGET_CLASSES = ("Liver", "Lung", "Spleen", "None")
IONIZABLE_SMILES_COLUMN = "Ionizable lipid(S)"


def data_load(file_path):
    """加载 .npy / .npz / .csv 数据。"""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = file_path.suffix.lower()
    if ext == ".npy":
        return np.load(file_path)
    if ext == ".npz":
        with np.load(file_path) as loaded:
            keys = list(loaded.keys())
            return loaded["arr_0"] if "arr_0" in keys else loaded[keys[0]]
    if ext == ".csv":
        return pd.read_csv(file_path).values.astype(np.float32)

    raise ValueError(f"不支持的文件格式: {file_path.suffix}")


def _clean_target_series(series):
    return (
        series.fillna("None")
        .astype(str)
        .str.strip()
        .replace({"nan": "None", "NaN": "None"})
    )


def _load_aligned_label_table(
    data_csv,
    embedding_npy,
    target_column=TARGET_COLUMN,
    target_classes=TARGET_CLASSES,
):
    """
    仅为“稳定对齐样本顺序”而加载 embedding。

    这个脚本只负责划分数据集，不做模型训练；但为了与后续 embedding 数组切片保持一致，
    仍然严格复用“按 CSV 与 embedding 最小长度对齐”的规则，避免索引错位。
    """
    data_csv = Path(data_csv)
    embedding_npy = Path(embedding_npy)

    df = pd.read_csv(data_csv)
    x_all = data_load(embedding_npy)
    if x_all.ndim != 2:
        raise ValueError(f"embedding 必须是二维数组，当前 shape={x_all.shape}")

    aligned_rows = min(len(df), len(x_all))
    if len(df) != len(x_all):
        warnings.warn(
            "CSV 行数与 embedding 行数不一致，已按最小长度对齐截断。"
            f" CSV={len(df)}, embedding={len(x_all)}, aligned={aligned_rows}"
        )

    df = df.iloc[:aligned_rows].copy()
    df["_aligned_index"] = np.arange(aligned_rows)

    df[target_column] = _clean_target_series(df[target_column])
    mask = df[target_column].isin(target_classes)
    df_filtered = df.loc[mask].copy().reset_index(drop=True)
    df_filtered["_filtered_index"] = np.arange(len(df_filtered))

    if len(df_filtered) == 0:
        raise ValueError("过滤目标标签后没有可用样本。")

    return df_filtered


def _subset_sum_cluster_split(group_sizes, target_size, seed=42, fixed_train_count=0):
    """
    在不打散簇的前提下，用子集和 DP 逼近目标测试集大小。
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


def _project_to_2d(features, seed=42, prefer_umap=True, metric="jaccard"):
    """优先使用 UMAP，可用性不足时自动回退到 PCA。针对化学指纹默认使用 jaccard (即 Tanimoto 距离)。"""
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
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"UMAP 不可用，自动回退到 PCA。原因: {exc}")

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


def _build_cluster_summary(df_split, unique_smiles_df, invalid_count):
    cluster_summary = (
        df_split.loc[df_split["_cluster_id"] >= 0]
        .groupby("_cluster_id")
        .agg(
            sample_count=("_filtered_index", "size"),
            y_unique_count=(TARGET_COLUMN, "nunique"),
        )
        .reset_index()
        .rename(columns={"_cluster_id": "cluster_id"})
    )

    unique_counts = (
        unique_smiles_df.groupby("cluster_id")
        .size()
        .reset_index(name="unique_lipid_count")
    )
    split_map = (
        df_split.loc[df_split["_cluster_id"] >= 0, ["_cluster_id", "_split"]]
        .drop_duplicates()
        .rename(columns={"_cluster_id": "cluster_id", "_split": "split"})
    )

    cluster_summary = cluster_summary.merge(unique_counts, on="cluster_id", how="left")
    cluster_summary = cluster_summary.merge(split_map, on="cluster_id", how="left")
    cluster_summary["is_test_cluster"] = cluster_summary["split"].eq("test")
    cluster_summary["is_val_cluster"] = cluster_summary["split"].eq("val")
    cluster_summary = cluster_summary.sort_values(
        by=["is_test_cluster", "is_val_cluster", "sample_count", "cluster_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    if invalid_count > 0:
        invalid_row = pd.DataFrame(
            [
                {
                    "cluster_id": -1,
                    "sample_count": int(invalid_count),
                    "y_unique_count": int(
                        df_split.loc[df_split["_cluster_id"] < 0, TARGET_COLUMN].nunique()
                    ),
                    "unique_lipid_count": 0,
                    "split": "train",
                    "is_test_cluster": False,
                    "is_val_cluster": False,
                }
            ]
        )
        cluster_summary = pd.concat([cluster_summary, invalid_row], ignore_index=True)

    return cluster_summary


def _save_projection_points(plot_df, coordinate_path, projection_method):
    coordinate_path = Path(coordinate_path)
    coordinate_path.parent.mkdir(parents=True, exist_ok=True)

    coordinate_df = plot_df.copy()
    coordinate_df["projection_method"] = projection_method
    selected_cols = ["canonical_smiles", "cluster_id", "split", "x", "y", "projection_method"]
    coordinate_df = coordinate_df[selected_cols].copy()
    coordinate_df.to_csv(coordinate_path, index=False, encoding="utf-8-sig")
    return coordinate_path


def _save_split_visualization(
    unique_smiles_df,
    fingerprint_matrix,
    cluster_summary,
    test_cluster_ids,
    figure_path,
    cutoff,
    seed=42,
    coordinate_path=None,
    val_cluster_ids=None,
    projection_metric="jaccard",
):
    import matplotlib.pyplot as plt

    val_cluster_ids = set(val_cluster_ids or [])
    test_cluster_ids = set(test_cluster_ids or [])

    projection, method_name = _project_to_2d(
        fingerprint_matrix, seed=seed, prefer_umap=True, metric=projection_metric
    )
    plot_df = unique_smiles_df.copy()
    plot_df["x"] = projection[:, 0]
    plot_df["y"] = projection[:, 1]
    if coordinate_path is not None:
        _save_projection_points(plot_df, coordinate_path, method_name)

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(17, 7),
        gridspec_kw={"width_ratios": [2.4, 1.1]},
    )

    scatter_ax, bar_ax = axes
    style_map = {
        "train": {"color": "#1f77b4", "marker": "o", "label": "Train"},
        "val": {"color": "#2ca02c", "marker": "s", "label": "Val"},
        "test": {"color": "#d62728", "marker": "^", "label": "Test"},
    }

    for split_name, style in style_map.items():
        mask = plot_df["split"].eq(split_name)
        if not mask.any():
            continue
        edgecolors = "black" if split_name in {"test", "val"} else style["color"]
        scatter_ax.scatter(
            plot_df.loc[mask, "x"],
            plot_df.loc[mask, "y"],
            c=style["color"],
            marker=style["marker"],
            s=56,
            alpha=0.82,
            edgecolors=edgecolors,
            linewidths=0.4,
            label=f"{style['label']} unique lipids ({int(mask.sum())})",
        )

    centroid_df = (
        plot_df.groupby("cluster_id")[["x", "y"]]
        .mean()
        .reset_index()
    )
    for _, row in centroid_df.iterrows():
        cluster_id = int(row["cluster_id"])
        if cluster_id in test_cluster_ids:
            text_color = "#d62728"
        elif cluster_id in val_cluster_ids:
            text_color = "#2ca02c"
        else:
            text_color = "#2f2f2f"
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

    title_parts = [f"Total clusters={len(centroid_df)}"]
    if val_cluster_ids:
        title_parts.append(f"Val clusters={sorted(val_cluster_ids)}")
    title_parts.append(f"Test clusters={sorted(test_cluster_ids)}")
    scatter_ax.set_title(
        f"Butina 2D Projection ({method_name})\n" + ", ".join(title_parts)
    )
    scatter_ax.set_xlabel("Component 1")
    scatter_ax.set_ylabel("Component 2")
    scatter_ax.legend(loc="best")

    bar_df = cluster_summary.copy()
    bar_df["cluster_label"] = bar_df["cluster_id"].apply(
        lambda value: "Invalid" if int(value) < 0 else f"C{int(value)}"
    )
    bar_df = bar_df.sort_values(by=["cluster_id"]).reset_index(drop=True)
    bar_colors = bar_df["split"].map({"train": "#1f77b4", "val": "#2ca02c", "test": "#d62728"}).fillna("#7f7f7f")

    bar_ax.barh(bar_df["cluster_label"], bar_df["sample_count"], color=bar_colors)
    for _, row in bar_df.iterrows():
        bar_ax.text(
            row["sample_count"] + 0.3,
            row["cluster_label"],
            str(int(row["sample_count"])),
            va="center",
            fontsize=8.5,
        )

    bar_ax.set_title(f"Cluster Size Distribution (cutoff={cutoff})")
    bar_ax.set_xlabel("Sample Count")
    bar_ax.set_ylabel("Cluster ID")

    fig.suptitle("Butina HPO Split Overview", fontsize=14, y=1.02)
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
    y_val_path = output_dir / f"{save_prefix}_y_val.csv"
    y_test_path = output_dir / f"{save_prefix}_y_test.csv"
    cluster_path = output_dir / f"{save_prefix}_cluster_summary.csv"
    summary_path = output_dir / f"{save_prefix}_summary.json"

    split_table.to_csv(split_path, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(cluster_path, index=False, encoding="utf-8-sig")

    train_table = split_table.loc[split_table["split"].eq("train"), ["filtered_index", "aligned_index", TARGET_COLUMN]]
    val_table = split_table.loc[split_table["split"].eq("val"), ["filtered_index", "aligned_index", TARGET_COLUMN]]
    test_table = split_table.loc[split_table["split"].eq("test"), ["filtered_index", "aligned_index", TARGET_COLUMN]]
    train_table.to_csv(y_train_path, index=False, encoding="utf-8-sig")
    if len(val_table) > 0:
        val_table.to_csv(y_val_path, index=False, encoding="utf-8-sig")
    test_table.to_csv(y_test_path, index=False, encoding="utf-8-sig")

    payload = dict(summary)
    payload["figure_path"] = str(figure_path) if figure_path else None
    payload["projection_points_path"] = str(coordinate_path) if coordinate_path else None
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return {
        "split_path": str(split_path),
        "y_train_path": str(y_train_path),
        "y_val_path": str(y_val_path) if len(val_table) > 0 else None,
        "y_test_path": str(y_test_path),
        "cluster_summary_path": str(cluster_path),
        "summary_path": str(summary_path),
        "figure_path": str(figure_path) if figure_path else None,
        "projection_points_path": str(coordinate_path) if coordinate_path else None,
    }


def get_butina_hpo_split(
    data_csv=DEFAULT_DATA_CSV,
    embedding_npy=DEFAULT_EMBEDDING_NPY,
    target_column=TARGET_COLUMN,
    target_classes=TARGET_CLASSES,
    ionizable_smiles_column=IONIZABLE_SMILES_COLUMN,
    radius=2,
    n_bits=2048,
    cutoff=0.6,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    visualize=False,
    output_dir=None,
    save_prefix=None,
    projection_metric="jaccard",
    morgan_fps_path=None,
    verbose=True,
):
    """
    基于 Butina 聚类的数据集划分（纯单簇独立背包划分），供超参数优化阶段复用。
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "运行 Butina 超参优化脚本需要安装 RDKit。"
            "可使用 conda install -c conda-forge rdkit"
        ) from exc

    df = _load_aligned_label_table(
        data_csv=data_csv,
        embedding_npy=embedding_npy,
        target_column=target_column,
        target_classes=target_classes,
    )

    from PCA_transductive_butina_split import _load_or_compute_morgan_fingerprints

    (
        canonical_smiles,
        unique_smiles,
        fingerprints,
        distance_matrix,
        fingerprint_matrix,
    ) = _load_or_compute_morgan_fingerprints(
        df_filtered=df,
        ionizable_smiles_column=ionizable_smiles_column,
        radius=radius,
        n_bits=n_bits,
        morgan_fps_path=morgan_fps_path,
        data_csv=data_csv,
        verbose=verbose,
    )

    df = df.copy()
    df["_canonical_smiles"] = canonical_smiles

    valid_mask = df["_canonical_smiles"].notna()
    invalid_count = int((~valid_mask).sum())
    if valid_mask.sum() == 0:
        raise ValueError("没有可用于 Butina 聚类的有效可电离脂质 SMILES。")

    clusters = Butina.ClusterData(
        distance_matrix,
        len(fingerprints),
        cutoff,
        isDistData=True,
    )
    if len(clusters) == 0:
        raise ValueError("Butina 未产生任何簇，请检查输入或调整 cutoff。")

    smiles_to_cluster = {}
    for cluster_id, member_indices in enumerate(clusters):
        for member_idx in member_indices:
            smiles_to_cluster[unique_smiles[member_idx]] = cluster_id

    df["_cluster_id"] = df["_canonical_smiles"].map(smiles_to_cluster).fillna(-1).astype(int)

    cluster_sizes = (
        df.loc[df["_cluster_id"] >= 0]
        .groupby("_cluster_id")
        .size()
        .astype(int)
        .to_dict()
    )

    target_test_size = max(1, int(round(len(df) * float(test_ratio))))
    target_val_size = (
        max(1, int(round(len(df) * float(val_ratio))))
        if (val_ratio and float(val_ratio) > 0.0)
        else 0
    )

    test_cluster_ids = _subset_sum_cluster_split(
        cluster_sizes,
        target_size=target_test_size,
        seed=seed,
        fixed_train_count=invalid_count,
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
                fixed_train_count=invalid_count,
            )

    df["_split"] = "train"
    df.loc[df["_cluster_id"].isin(val_cluster_ids), "_split"] = "val"
    df.loc[df["_cluster_id"].isin(test_cluster_ids), "_split"] = "test"
    df.loc[df["_cluster_id"] < 0, "_split"] = "train"

    # 防止数据泄露：同一 canonical SMILES 绝不能跨 split。
    split_per_smiles = (
        df.loc[df["_cluster_id"] >= 0, ["_canonical_smiles", "_split"]]
        .drop_duplicates()
        .groupby("_canonical_smiles")["_split"]
        .nunique()
    )
    if int(split_per_smiles.max()) > 1:
        raise RuntimeError("检测到同一可电离脂质跨越多个划分集，划分失败。")

    train_mask = df["_split"].eq("train").to_numpy()
    val_mask = df["_split"].eq("val").to_numpy()
    test_mask = df["_split"].eq("test").to_numpy()
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Butina 划分后训练集或测试集为空，请调整 cutoff 或 test_ratio。")
    if val_ratio and float(val_ratio) > 0.0 and val_mask.sum() == 0:
        raise ValueError("Butina 划分后验证集为空，请调整 cutoff 或 val_ratio。")

    unique_smiles_df = pd.DataFrame(
        {
            "canonical_smiles": unique_smiles,
            "cluster_id": [smiles_to_cluster[smiles] for smiles in unique_smiles],
        }
    )
    conditions = [
        unique_smiles_df["cluster_id"].isin(list(test_cluster_ids)),
        unique_smiles_df["cluster_id"].isin(list(val_cluster_ids)),
    ]
    choices = ["test", "val"]
    unique_smiles_df["split"] = np.select(conditions, choices, default="train")

    cluster_summary = _build_cluster_summary(
        df,
        unique_smiles_df,
        invalid_count=invalid_count,
    )

    split_table = df.rename(
        columns={
            "_filtered_index": "filtered_index",
            "_aligned_index": "aligned_index",
            "_cluster_id": "cluster_id",
            "_split": "split",
            "_canonical_smiles": "canonical_smiles",
        }
    )[
        [
            "filtered_index",
            "aligned_index",
            target_column,
            "split",
            "cluster_id",
            ionizable_smiles_column,
            "canonical_smiles",
        ]
    ].copy()

    split_table["is_test_cluster"] = split_table["cluster_id"].isin(list(test_cluster_ids))
    split_table["is_val_cluster"] = split_table["cluster_id"].isin(list(val_cluster_ids))
    split_table["is_valid_smiles"] = split_table["cluster_id"].ge(0)

    figure_path = None
    projection_points_path = None
    if visualize:
        output_base_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        file_stem = save_prefix or f"butina_cutoff{cutoff}_seed{seed}"
        figure_path = output_base_dir / f"{file_stem}_visualization.png"
        projection_points_path = output_base_dir / f"{file_stem}_projection_points.csv"
        figure_path = _save_split_visualization(
            unique_smiles_df=unique_smiles_df,
            fingerprint_matrix=fingerprint_matrix,
            cluster_summary=cluster_summary,
            test_cluster_ids=test_cluster_ids,
            val_cluster_ids=val_cluster_ids,
            figure_path=figure_path,
            cutoff=cutoff,
            seed=seed,
            coordinate_path=projection_points_path,
            projection_metric=projection_metric,
        )

    summary = {
        "method": "butina",
        "seed": int(seed),
        "cutoff": float(cutoff),
        "radius": int(radius),
        "n_bits": int(n_bits),
        "total_samples": int(len(df)),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "val_ratio_target": float(val_ratio) if val_ratio else 0.0,
        "test_ratio_target": float(test_ratio),
        "val_ratio_actual": float(val_mask.sum() / len(df)),
        "test_ratio_actual": float(test_mask.sum() / len(df)),
        "total_clusters": int(len(clusters)),
        "test_cluster_ids": sorted(int(x) for x in test_cluster_ids),
        "val_cluster_ids": sorted(int(x) for x in val_cluster_ids),
        "train_cluster_ids": sorted(
            int(x) for x in cluster_summary.loc[
                cluster_summary["cluster_id"].ge(0) & cluster_summary["split"].eq("train"),
                "cluster_id",
            ].tolist()
        ),
        "invalid_smiles_count": int(invalid_count),
        "y_train_distribution": pd.Series(df.loc[train_mask, target_column]).value_counts().to_dict(),
        "y_val_distribution": pd.Series(df.loc[val_mask, target_column]).value_counts().to_dict(),
        "y_test_distribution": pd.Series(df.loc[test_mask, target_column]).value_counts().to_dict(),
    }

    saved_paths = None
    if output_dir is not None:
        saved_paths = _save_outputs(
            output_dir=output_dir,
            save_prefix=save_prefix or f"butina_cutoff{cutoff}_seed{seed}",
            split_table=split_table,
            cluster_summary=cluster_summary,
            summary=summary,
            figure_path=figure_path,
            coordinate_path=projection_points_path,
        )

    if verbose:
        print("=" * 90)
        print("[Butina-HPO] 数据集划分完成 (策略: cluster)")
        print(f"[Butina-HPO] 总样本数: {summary['total_samples']}")
        print(f"[Butina-HPO] 总簇数: {summary['total_clusters']}")
        print(f"[Butina-HPO] 验证簇: {summary['val_cluster_ids']}")
        print(f"[Butina-HPO] 测试簇: {summary['test_cluster_ids']}")
        print(f"[Butina-HPO] 训练集样本数: {summary['train_samples']} ({summary['train_samples']/len(df):.1%})")
        print(f"[Butina-HPO] 验证集样本数: {summary['val_samples']} ({summary['val_ratio_actual']:.1%})")
        print(f"[Butina-HPO] 测试集样本数: {summary['test_samples']} ({summary['test_ratio_actual']:.1%})")
        print(f"[Butina-HPO] y_train 分布: {summary['y_train_distribution']}")
        print(f"[Butina-HPO] y_val 分布: {summary['y_val_distribution']}")
        print(f"[Butina-HPO] y_test 分布: {summary['y_test_distribution']}")
        if saved_paths:
            print(f"[Butina-HPO] 输出目录: {output_dir}")

    return {
        "train_indices": split_table.loc[split_table["split"].eq("train"), "filtered_index"].to_numpy(),
        "val_indices": split_table.loc[split_table["split"].eq("val"), "filtered_index"].to_numpy(),
        "test_indices": split_table.loc[split_table["split"].eq("test"), "filtered_index"].to_numpy(),
        "train_aligned_indices": split_table.loc[split_table["split"].eq("train"), "aligned_index"].to_numpy(),
        "val_aligned_indices": split_table.loc[split_table["split"].eq("val"), "aligned_index"].to_numpy(),
        "test_aligned_indices": split_table.loc[split_table["split"].eq("test"), "aligned_index"].to_numpy(),
        "y_train": split_table.loc[split_table["split"].eq("train"), target_column].to_numpy(),
        "y_val": split_table.loc[split_table["split"].eq("val"), target_column].to_numpy(),
        "y_test": split_table.loc[split_table["split"].eq("test"), target_column].to_numpy(),
        "split_table": split_table,
        "cluster_summary": cluster_summary,
        "val_cluster_ids": sorted(int(x) for x in val_cluster_ids),
        "test_cluster_ids": sorted(int(x) for x in test_cluster_ids),
        "summary": summary,
        "saved_paths": saved_paths,
        "figure_path": str(figure_path) if figure_path else None,
        "projection_points_path": str(projection_points_path) if projection_points_path else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="基于 Butina 的超参数优化数据集划分脚本")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV), help="标签 CSV 路径")
    parser.add_argument(
        "--embedding_npy",
        type=str,
        default=str(DEFAULT_EMBEDDING_NPY),
        help="用于对齐样本顺序的 embedding 路径",
    )
    parser.add_argument("--cutoff", type=float, default=0.6, help="Butina 距离 cutoff")
    parser.add_argument("--radius", type=int, default=2, help="Morgan 指纹半径")
    parser.add_argument("--n_bits", type=int, default=2048, help="Morgan 指纹位数")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="目标验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="目标测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="划分结果输出目录",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="",
        help="输出文件名前缀，默认按参数自动生成",
    )
    parser.add_argument(
        "--projection_metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "euclidean"],
        help="2D 降维距离度量，默认 jaccard (即化学分子 Tanimoto 距离)",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="仅输出划分结果，不保存可视化图片",
    )
    parser.add_argument(
        "--morgan_fps_path",
        type=str,
        default="",
        help="预计算的 Morgan 指纹 .pkl 文件路径（留空则自动检测数据集同级目录下的 *_morgan_fingerprints.pkl）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = get_butina_hpo_split(
        data_csv=args.data_csv,
        embedding_npy=args.embedding_npy,
        radius=args.radius,
        n_bits=args.n_bits,
        cutoff=args.cutoff,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        morgan_fps_path=args.morgan_fps_path,
        visualize=not args.no_visualize,
        output_dir=args.output_dir,
        save_prefix=args.save_prefix or None,
        projection_metric=args.projection_metric,
        verbose=True,
    )

    print("=" * 90)
    print("[Butina-HPO] 独立运行摘要")
    print(f"[Butina-HPO] y_train 前10项: {result['y_train'][:10].tolist()}")
    if len(result['y_val']) > 0:
        print(f"[Butina-HPO] y_val 前10项: {result['y_val'][:10].tolist()}")
    print(f"[Butina-HPO] y_test 前10项: {result['y_test'][:10].tolist()}")
