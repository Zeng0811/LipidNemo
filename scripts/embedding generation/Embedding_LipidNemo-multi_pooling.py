#!/usr/bin/env python
# coding: utf-8
"""
This script extracts multi-pooling representations (mean, max, cls, sum)
from SMILES sequences of lipid nanoparticle (LNP) formulation components
using NVIDIA BioNeMo MegaMolBART models, and concatenates them with
normalized component molar ratios.
"""

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import warnings
import numpy as np
import pandas as pd
from pytorch_lightning import Trainer
import torch
import torch.distributed as dist
import yaml

# ----------------- 1. BioNeMo Environment Check -----------------
def check_bionemo_env(bionemo_dir="/workspace/bionemo"):
    """Verify that BioNeMo and Megatron Core are accessible."""
    if os.path.exists(bionemo_dir) and bionemo_dir not in sys.path:
        sys.path.append(bionemo_dir)

    try:
        from bionemo.model.molecule.megamolbart.megamolbart_model import (
            MegaMolBARTModel,
        )
        from megatron.core import parallel_state

        print("[INFO] Environment check passed: BioNeMo / Megatron Core")
        return MegaMolBARTModel, parallel_state
    except ImportError as e:
        sys.exit(
            f"[ERROR] Missing dependency: Please execute inside the NVIDIA BioNeMo container. "
            f"Details: {e}"
        )


warnings.filterwarnings("ignore")

# ----------------- 2. Formulation Component Configuration -----------------
# Mapping of component SMILES column name -> Molar ratio column name
COMPONENT_CONFIG = {
    "Ionizable lipid(S)": "Ionizable lipid(M)",
    "Helper lipid(S)": "Helper lipid(M)",
    "Cholesterol(S)": "Cholesterol(M)",
    "PEG(S)": "PEG(M)",
    "Additional component(S)": "Additional component(M)",
}

# ----------------- 3. Distributed Environment Setup -----------------
def setup_env(parallel_state):
    """Initialize distributed process group and Megatron parallel state."""
    # Set default distributed environment variables for single-GPU inference
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("EXPERT_PARALLEL", "0")
    os.environ.setdefault("MODEL_PARALLEL", "0")
    os.environ.setdefault("PIPELINE_PARALLEL", "0")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    try:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
    except Exception:
        pass

    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            context_parallel_size=1,
        )
    except Exception as e:
        print(f"[ERROR] Megatron initialization failed: {e}")
        sys.exit(1)


# ----------------- 4. Model Loading with Config Override -----------------
def load_model(model_cls, model_path):
    """
    Load MegaMolBART model from checkpoint with runtime config overrides.
    Extracts and adjusts model_config.yaml dynamically to guarantee stable inference.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint file not found: {model_path}")
        sys.exit(1)

    print(f"[MODEL] Loading model checkpoint: {os.path.basename(model_path)}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract model config from .nemo tar archive
        with tarfile.open(model_path, "r:") as tar:
            member = next(
                m for m in tar.getmembers() if "model_config.yaml" in m.name
            )
            tar.extract(member, path=temp_dir)

        config_path = os.path.join(temp_dir, "model_config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Override batch sizes for inference
        config["global_batch_size"] = 1
        config["micro_batch_size"] = 1

        # Configuration patch to disable training-specific data augmentation
        if "data" not in config:
            config["data"] = {}
        config["data"]["decoder_independent_augment"] = False
        config["data"]["canonicalize_target_smile"] = False
        config["data"]["canonicalize_decoder_output"] = False
        config["data"]["canonicalize_encoder_input"] = False

        override_path = os.path.join(temp_dir, "override_config.yaml")
        with open(override_path, "w") as f:
            yaml.dump(config, f)

        trainer = Trainer(accelerator="gpu", devices=[0], logger=False)
        model = model_cls.restore_from(
            restore_path=model_path,
            trainer=trainer,
            strict=False,
            override_config_path=override_path,
        )

    model.eval().cuda()
    return model


# ----------------- 5. Pooling Functions -----------------
def apply_pooling(encoder_outputs, attn_mask, method):
    """
    Perform mathematical pooling on hidden states.

    Args:
        encoder_outputs (torch.Tensor): Output tensor of shape (Batch, Seq_Len, Hidden_Dim).
        attn_mask (torch.Tensor): Attention mask tensor of shape (Batch, Seq_Len).
        method (str): Pooling strategy, one of ['mean', 'max', 'cls', 'sum'].

    Returns:
        torch.Tensor: Pooled representation of shape (Batch, Hidden_Dim).
    """
    mask = attn_mask.unsqueeze(-1)  # (Batch, Seq_Len, 1)

    if method == "mean":
        # Masked average pooling
        return torch.sum(encoder_outputs * mask, dim=1) / (
            torch.sum(mask, dim=1) + 1e-9
        )

    elif method == "max":
        # Masked max pooling: mask padding tokens with boundary value
        fill_val = (
            -float("inf")
            if encoder_outputs.dtype in (torch.float16, torch.bfloat16)
            else -1e9
        )
        masked_outputs = encoder_outputs.masked_fill(mask == 0, fill_val)
        max_val, _ = torch.max(masked_outputs, dim=1)
        return max_val

    elif method in ("cls", "bos", "first"):
        # First token representation (index 0)
        return encoder_outputs[:, 0, :]

    elif method == "sum":
        # Masked sum pooling
        return torch.sum(encoder_outputs * mask, dim=1)

    else:
        raise ValueError(f"[ERROR] Unsupported pooling method: {method}")


def get_embeddings_multi_pooling(
    model, smiles_list, pooling_methods, batch_size=64
):
    """
    Extract multi-pooling embeddings for a single component list via single forward pass.
    """
    total_len = len(smiles_list)
    valid_indices, valid_smiles = [], []

    for i, s in enumerate(smiles_list):
        if pd.isna(s):
            continue
        s_str = str(s).strip()

        # Filter empty entries and placeholder slashes while preserving stereochemical bonds
        if (
            s_str == ""
            or s_str.lower() == "nan"
            or s_str in ("\\", "\\\\", "/", "//")
        ):
            continue

        # Remove accidental trailing backslashes if present
        if s_str.endswith("\\") and not s_str.endswith("\\\\"):
            s_str = s_str.rstrip("\\").strip()
            if s_str == "":
                continue

        valid_indices.append(i)
        valid_smiles.append(s_str)

    # Initialize output dictionary with zeros (512-dim embedding per component)
    results = {
        m: np.zeros((total_len, 512), dtype=np.float32) for m in pooling_methods
    }
    if not valid_smiles:
        return results

    tokenizer = model.tokenizer
    batch_pooled_results = {m: [] for m in pooling_methods}

    # Mini-batch inference to prevent GPU memory overflow
    for start_idx in range(0, len(valid_smiles), batch_size):
        end_idx = min(start_idx + batch_size, len(valid_smiles))
        batch_smiles = valid_smiles[start_idx:end_idx]

        token_ids = [tokenizer.text_to_ids(s) for s in batch_smiles]
        max_len = max(len(ids) for ids in token_ids)

        input_ids = torch.full(
            (len(batch_smiles), max_len),
            tokenizer.pad_id,
            dtype=torch.long,
            device="cuda:0",
        )
        attn_mask = torch.zeros(
            (len(batch_smiles), max_len), dtype=torch.long, device="cuda:0"
        )

        for i, ids in enumerate(token_ids):
            input_ids[i, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device="cuda:0"
            )
            attn_mask[i, : len(ids)] = 1

        with torch.no_grad():
            encoder_outputs = model.encode(input_ids, attn_mask)

            # Compute representations across all pooling methods from shared hidden states
            for m in pooling_methods:
                pooled = apply_pooling(encoder_outputs, attn_mask, m)
                batch_pooled_results[m].append(pooled.cpu().numpy())

    # Concatenate batches and map back to corresponding original indices
    for m in pooling_methods:
        if batch_pooled_results[m]:
            concatenated = np.vstack(batch_pooled_results[m])
            results[m][valid_indices] = concatenated

    return results


# ----------------- 6. Feature Comparison & Validation -----------------
def compare_pooling_embeddings(output_dir, prefix, methods):
    """Verify consistency and distinctiveness across pooling representations."""
    print("\n" + "=" * 60)
    print("[COMPARE] Cross-validation of pooled feature matrices...")
    print("=" * 60)
    loaded = {}
    for m in methods:
        p = os.path.join(output_dir, f"{prefix}-{m}.npy")
        if os.path.exists(p):
            loaded[m] = np.load(p)
            print(
                f"   [LOAD] [{m:5s}] Shape: {loaded[m].shape} | "
                f"Mean: {loaded[m].mean():.4f} | Std: {loaded[m].std():.4f}"
            )

    keys = list(loaded.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            m1, m2 = keys[i], keys[j]
            # Exclude trailing 5 molar ratio dimensions to compare only molecular features (2560-dim)
            e1 = loaded[m1][:, :-5]
            e2 = loaded[m2][:, :-5]

            norm1 = np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9
            norm2 = np.linalg.norm(e2, axis=1, keepdims=True) + 1e-9
            cos_sim = np.sum((e1 / norm1) * (e2 / norm2), axis=1).mean()
            diff_l2 = np.linalg.norm(e1 - e2, axis=1).mean()

            is_identical = np.array_equal(e1, e2)
            status_text = (
                "[WARN] Identical representations detected"
                if is_identical
                else "[OK] Distinct representations verified"
            )

            print(f"\n[COMPARE] [{m1:5s}] vs [{m2:5s}]:")
            print(f"   - Mean Cosine Similarity : {cos_sim:.4f}")
            print(f"   - Mean L2 Distance       : {diff_l2:.4f}")
            print(f"   - Status                 : {status_text}")
    print("=" * 60)


# ----------------- 7. Main Pipeline -----------------
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="BioNeMo MegaMolBART Multi-Pooling Feature Extraction for LNP Formulations"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data/lnp_formulations.csv",
        help="Path to formulation CSV file (default: './data/lnp_formulations.csv')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/MegaMolBART_ft.nemo",
        help="Path to BioNeMo checkpoint file (.nemo) (default: './models/MegaMolBART_ft.nemo')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/pooling_methods",
        help="Directory to save output feature matrices (default: './results/pooling_methods')",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="vector_concat_ft",
        help="Prefix for saved .npy files (default: 'vector_concat_ft')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64)",
    )
    parser.add_argument(
        "--pooling-methods",
        nargs="+",
        default=["mean", "max", "cls", "sum"],
        help="List of pooling strategies to apply (default: mean max cls sum)",
    )
    parser.add_argument(
        "--bionemo-dir",
        type=str,
        default=os.environ.get("BIONEMO_DIR", "/workspace/bionemo"),
        help="BioNeMo root directory inside the environment (default: '/workspace/bionemo')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index to run inference on (default: '0')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Configure device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # 1. Environment check
    MegaMolBARTModel, parallel_state = check_bionemo_env(args.bionemo_dir)

    print("=" * 60)
    print("[START] Multi-pooling feature generation pipeline")
    print(f"[DATA] Dataset path: {args.data_path}")
    print(f"[INFO] Pooling methods: {args.pooling_methods}")
    print(f"[MODEL] Model path: {args.model_path}")
    print(f"[SAVE] Output directory: {args.output_dir}")
    print("=" * 60)

    if not os.path.exists(args.data_path):
        sys.exit(
            f"[ERROR] Data file not found: {args.data_path}. "
            f"Please verify path or specify via --data-path."
        )

    # 2. Load dataset and clean invalid placeholder characters
    df = pd.read_csv(args.data_path)
    df.replace(
        to_replace=[r"^\s*\\+\s*$", r"^\s*/*\s*$", r"^\s*$"],
        value=np.nan,
        regex=True,
        inplace=True,
    )
    n_samples = len(df)
    print(f"[DATA] Formulation dataset loaded. Valid sample count: {n_samples}")

    setup_env(parallel_state)
    model = load_model(MegaMolBARTModel, args.model_path)

    pooled_embs_by_method = {m: [] for m in args.pooling_methods}
    ratio_list = []

    # 3. Process each formulation component
    for comp_idx, (smi_col, ratio_col) in enumerate(
        COMPONENT_CONFIG.items(), start=1
    ):
        print(
            f"\n[DATA] [{comp_idx}/{len(COMPONENT_CONFIG)}] Processing component pair: [{smi_col}] + [{ratio_col}]"
        )

        # Extract embeddings for requested pooling methods
        if smi_col not in df.columns:
            print(f"   [WARN] Column '{smi_col}' not found, filling with zero vectors.")
            for m in args.pooling_methods:
                pooled_embs_by_method[m].append(
                    np.zeros((n_samples, 512), dtype=np.float32)
                )
        else:
            comp_results = get_embeddings_multi_pooling(
                model=model,
                smiles_list=df[smi_col].tolist(),
                pooling_methods=args.pooling_methods,
                batch_size=args.batch_size,
            )
            for m in args.pooling_methods:
                pooled_embs_by_method[m].append(comp_results[m])
            print(
                f"   [OK] Completed pooling representations for '{smi_col}' (512-dim per method)"
            )

        # Extract molar ratio and normalize to [0, 1]
        if ratio_col not in df.columns:
            ratios = np.zeros((n_samples, 1), dtype=np.float32)
        else:
            ratios = (
                pd.to_numeric(df[ratio_col], errors="coerce")
                .fillna(0)
                .values.reshape(-1, 1)
                .astype(np.float32)
                / 100.0
            )
        ratio_list.append(ratios)

    # Concatenate all component molar ratios -> shape (N, 5)
    all_rats = np.hstack(ratio_list)

    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Save combined feature matrices (2560 molecular dims + 5 molar ratio dims = 2565 dims)
    print("\n" + "=" * 60)
    print("[SAVE] Saving fused pooling feature matrices (.npy):")
    for m in args.pooling_methods:
        all_embs = np.hstack(pooled_embs_by_method[m])  # (N, 2560)
        x_final = np.hstack([all_embs, all_rats])      # (N, 2565)

        save_filename = f"{args.output_prefix}-{m}.npy"
        save_path = os.path.join(args.output_dir, save_filename)
        np.save(save_path, x_final)
        print(f"   [SAVE] [{m:5s}] -> Shape: {x_final.shape} | Saved to: {save_path}")
    print("=" * 60)

    # 5. Release distributed resources
    try:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    # 6. Feature validation and differentiation report
    compare_pooling_embeddings(args.output_dir, args.output_prefix, args.pooling_methods)
    print("\n[DONE] All multi-pooling feature matrices generated successfully.")


if __name__ == "__main__":
    main()
