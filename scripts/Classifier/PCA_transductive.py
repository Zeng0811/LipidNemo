# Transductive learning: sequentially select one sample from external data to merge with train set for PCA
# Supports controllable validation/test set evaluation based on cluster OOD split (specified via --eval_split val/test)
# The training set is strictly used for feature fitting and model training, never evaluated for accuracy

import argparse
import copy
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


# Force HuggingFace offline mode, consistent with original script.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["AG_MAX_MEMORY_USAGE_RATIO"] = "100.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

TARGET_COLUMN = "Organ"
TARGET_CLASSES = ["Liver", "Lung", "Spleen", "None"]
RATIO_COLS = 5


# ================= transformer (Scratch) =================
class SimpleTransformerModule(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.squeeze_layer(x) if hasattr(self, 'squeeze_layer') else x.squeeze(1)
        return self.classifier(x)


class SklearnTransformer(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, num_classes=None, epochs=200, lr=0.0005, device='cuda:0', patience=15):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.patience = patience
        self.model = None
        self.label_encoder_ = None
        self.classes_ = None
        self.input_dim_ = None
        self.num_classes_ = None

    def _resolve_device(self):
        if isinstance(self.device, str) and self.device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] GPU unavailable, Transformer falling back to CPU.")
            return "cpu"
        return self.device

    def _build_model(self):
        self.model = SimpleTransformerModule(
            input_dim=self.input_dim_,
            num_classes=self.num_classes_,
        ).to(self._resolve_device())

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("[ERROR] Transformer input must be a 2D feature matrix.")
        if len(X) < 2:
            raise ValueError("[ERROR] Transformer requires at least 2 training samples.")

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        self.input_dim_ = X.shape[1] if self.input_dim is None else int(self.input_dim)
        self.num_classes_ = len(self.classes_) if self.num_classes is None else int(self.num_classes)

        device = self._resolve_device()
        self._build_model()

        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y_enc, dtype=torch.long))
        batch_size = max(4, min(32, len(X)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(loader))
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X):
        if self.model is None or self.label_encoder_ is None:
            raise ValueError("[ERROR] Model has not been fitted yet, cannot predict.")
        X = np.asarray(X, dtype=np.float32)
        device = self._resolve_device()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X, dtype=torch.float32).to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        pred_idx = np.argmax(probs, axis=1)
        return self.label_encoder_.inverse_transform(pred_idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state["_model_state_dict"] = {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            }
            state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model_state_dict = self.__dict__.pop("_model_state_dict", None)
        if model_state_dict is not None:
            self._build_model()
            self.model.load_state_dict(model_state_dict)
            self.model.eval()


# ================= TabPFN Wrapper =================
class TabPFNClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, device="cuda", n_estimators=8, max_time=120):
        self.device = device
        self.n_estimators = n_estimators
        self.max_time = max_time
        self.model_ = None
        self.classes_ = None

    def _resolve_device(self):
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("[WARN] GPU unavailable, TabPFN falling back to CPU.")
            return "cpu"
        return self.device

    def fit(self, X, y):
        device = self._resolve_device()
        # Prefer AutoTabPFNClassifier, fallback to TabPFNClassifier if unavailable
        try:
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
                AutoTabPFNClassifier,
            )
            self.model_ = AutoTabPFNClassifier(
                ignore_pretraining_limits=True,
                random_state=42,
                n_estimators=self.n_estimators,
                max_time=self.max_time,
                device=device,
            )
        except Exception:
            from tabpfn import TabPFNClassifier
            try:
                self.model_ = TabPFNClassifier(
                    device=device,
                    n_estimators=self.n_estimators,
                    random_state=42,
                    ignore_pretraining_limits=True,
                )
            except TypeError:
                try:
                    self.model_ = TabPFNClassifier(
                        device=device,
                        n_estimators=self.n_estimators,
                        random_state=42,
                    )
                except TypeError:
                    self.model_ = TabPFNClassifier(
                        device=device,
                        N_ensemble_configurations=self.n_estimators,
                        only_inference=True,
                        seed=42,
                    )
        self.model_.fit(X, y)
        self.classes_ = get_model_classes(self.model_)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise ValueError("[ERROR] TabPFN has not been fitted yet, cannot predict.")
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError("[ERROR] TabPFN has not been fitted yet, cannot predict.")
        return self.model_.predict_proba(X)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LNP sequential sample-by-sample transductive PCA and classification evaluation script based on cluster OOD split (supports controllable val/test set evaluation)"
    )

    # Data inputs
    parser.add_argument("--data_csv", type=str, required=True, help="Path to label CSV")
    parser.add_argument("--embedding_train", type=str, required=True, help="Path to training embedding .npy/.npz file")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save trained models")
    parser.add_argument("--new_data_npy", type=str, default="", help="Path to new sample .npy file for transductive PCA")
    parser.add_argument("--new_data_csv", type=str, default="", help="Optional path to new sample CSV (if containing Organ column, used to compute transductive accuracy)")

    # Run naming
    parser.add_argument("--embedding_name", type=str, default="Embedding", help="Embedding name used for output naming (default: Embedding)")
    parser.add_argument("--save_name", type=str, default="", help="Custom model save filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Evaluation target selection
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Subset used to evaluate model accuracy: val (validation set, default), test (test set)",
    )

    # Cluster split (pure Butina split)
    parser.add_argument(
        "--split_method",
        type=str,
        default="butina",
        choices=["butina"],
        help="Dataset split method, only 'butina' is supported (default: butina)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Target validation set ratio (default: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Target test set ratio (default: 0.1)")
    parser.add_argument(
        "--min_train_ratio",
        type=float,
        default=0.75,
        help="Minimum ratio of training samples to retain (default: 0.75)",
    )
    parser.add_argument(
        "--projection_metric",
        type=str,
        default="jaccard",
        choices=["jaccard", "euclidean"],
        help="Distance metric for Butina 2D projection (default: jaccard)",
    )
    parser.add_argument("--split_visualize", action="store_true", help="Whether to save cluster split visualization plot")

    # Butina parameters
    parser.add_argument("--butina_radius", type=int, default=2, help="Morgan fingerprint radius (default: 2)")
    parser.add_argument("--butina_n_bits", type=int, default=2048, help="Morgan fingerprint bit length (default: 2048)")
    parser.add_argument(
        "--butina_cutoff",
        "--cutoff",
        type=float,
        default=0.55,
        help="Butina distance cutoff (default: 0.55)",
    )
    parser.add_argument(
        "--morgan_fps_path",
        type=str,
        default="",
        help="Path to precomputed Morgan fingerprint .pkl file (leave empty to auto-detect *_morgan_fingerprints.pkl in dataset directory)",
    )

    # PCA parameters
    parser.add_argument("--pca_n", type=int, default=20, help="PCA dimension reduction target (<=0 disables PCA and uses raw embedding, default: 20)")
    parser.add_argument(
        "--inductive_embedding_mode",
        type=str,
        default="pca",
        choices=["pca", "raw"],
        help="Embedding mode for inductive learning: 'pca' or 'raw' (default: pca)",
    )

    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="tabpfn",
        choices=[
            "tabpfn",
            "LightGBM",
            "lightgbm",
            "lgbm",
            "random_forest",
            "rf",
            "Transformer",
            "transformer",
            "svm",
            "svc",
        ],
        help="Classifier type (default: tabpfn)",
    )

    # TabPFN
    parser.add_argument("--tabpfn_device", type=str, default="auto", help="TabPFN device (default: auto)")
    parser.add_argument("--tabpfn_n_estimators", type=int, default=8, help="TabPFN number of ensemble estimators (default: 8)")
    parser.add_argument("--tabpfn_max_time", type=int, default=120, help="TabPFN maximum inference time in seconds (default: 120)")

    # Random Forest
    parser.add_argument("--rf_n_estimators", type=int, default=100, help="Random Forest n_estimators (default: 100)")

    # LightGBM
    parser.add_argument("--lgbm_n_estimators", type=int, default=100, help="LightGBM n_estimators (default: 100)")
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05, help="LightGBM learning rate (default: 0.05)")
    parser.add_argument("--lgbm_num_leaves", type=int, default=31, help="LightGBM num_leaves (default: 31)")
    parser.add_argument("--lgbm_max_depth", type=int, default=-1, help="LightGBM max_depth (default: -1)")

    # Transformer
    parser.add_argument("--trans_epochs", type=int, default=200, help="Transformer training epochs (default: 200)")
    parser.add_argument("--trans_lr", type=float, default=0.0005, help="Transformer learning rate (default: 0.0005)")
    parser.add_argument("--trans_device", type=str, default="cuda:0", help="Transformer device (default: cuda:0)")
    parser.add_argument("--trans_patience", type=int, default=15, help="Transformer early stopping patience (default: 15)")

    # SVM
    parser.add_argument("--svm_c", type=float, default=1.0, help="SVM regularization parameter C (default: 1.0)")

    # Storage control
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="Do not save trained model weights .pkl file to save disk space",
    )

    return parser.parse_args()


def data_load(file_path):
    """Smart loader function, automatically identifies .npy / .npz / .csv."""
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    ext = file_path.suffix.lower()
    try:
        if ext == ".npy":
            data = np.load(file_path)
        elif ext == ".npz":
            with np.load(file_path) as loaded:
                keys = list(loaded.keys())
                data = loaded["arr_0"] if "arr_0" in keys else loaded[keys[0]]
        elif ext == ".csv":
            df = pd.read_csv(file_path, header=0)
            data = df.values.astype(np.float32)
        else:
            print(f"[WARN] Unsupported file format: {ext}")
            return None
        return data
    except Exception as exc:
        print(f"[ERROR] Failed to read: {exc}")
        return None


def _clean_target_series(series):
    return (
        series.fillna("None")
        .astype(str)
        .str.strip()
        .replace({"nan": "None", "NaN": "None"})
    )


def _choose_sample_name_series(df):
    candidate_columns = [
        "Sample",
        "sample",
        "Name",
        "name",
        "ID",
        "id",
        "LNP",
        "Formula",
        "Ionizable lipid(S)",
        "Ionizable lipid",
    ]
    for column in candidate_columns:
        if column in df.columns:
            return df[column].fillna("").astype(str).str.strip(), column

    if len(df.columns) > 0:
        first_column = df.columns[0]
        return df[first_column].fillna("").astype(str).str.strip(), first_column

    return pd.Series(dtype=str), ""


def load_new_data_bundle(new_data_npy, new_data_csv="", target_column=TARGET_COLUMN):
    """
    Load new sample data required for transductive prediction.
    If new_data_csv is provided, attempt to align sample names and true labels.
    """
    new_path = Path(new_data_npy)
    if not new_path.exists():
        raise FileNotFoundError(f"[ERROR] new_data_npy not found: {new_data_npy}")

    x_new_raw = data_load(new_path)
    if x_new_raw is None:
        raise ValueError(f"[ERROR] Failed to load new_data_npy: {new_data_npy}")
    if x_new_raw.ndim != 2:
        raise ValueError("[ERROR] new_data_npy must be a 2D array.")

    metadata_df = pd.DataFrame(index=np.arange(x_new_raw.shape[0]))
    if new_data_csv:
        csv_path = Path(new_data_csv)
        if csv_path.exists():
            info_df = pd.read_csv(csv_path)
            aligned_rows = min(len(info_df), len(x_new_raw))
            if len(info_df) != len(x_new_raw):
                print(
                    "[WARN] Row count mismatch between new_data_csv and new_data_npy, "
                    f"aligned to minimum length: CSV={len(info_df)}, npy={len(x_new_raw)}, aligned={aligned_rows}"
                )
            metadata_df = info_df.iloc[:aligned_rows].copy().reset_index(drop=True)
            x_new_raw = x_new_raw[:aligned_rows]
        else:
            print(f"[WARN] new_data_csv not found, defaulting to generated sample names: {new_data_csv}")

    metadata_df = metadata_df.reset_index(drop=True)
    if "sample_index" in metadata_df.columns:
        metadata_df = metadata_df.drop(columns=["sample_index"])
    metadata_df.insert(0, "sample_index", np.arange(len(x_new_raw), dtype=int))

    if len(metadata_df) > 0:
        sample_names, sample_name_source = _choose_sample_name_series(metadata_df)
    else:
        sample_names = pd.Series(dtype=str)
        sample_name_source = ""

    if len(sample_names) != len(x_new_raw):
        sample_names = pd.Series([f"Sample_{idx + 1}" for idx in range(len(x_new_raw))], dtype=str)
        sample_name_source = "generated"

    if "sample_name" in metadata_df.columns:
        metadata_df = metadata_df.drop(columns=["sample_name"])
    metadata_df.insert(1, "sample_name", sample_names.tolist())

    y_true = None
    if target_column in metadata_df.columns:
        y_true = _clean_target_series(metadata_df[target_column]).to_numpy()

    x_new_raw = np.nan_to_num(
        np.asarray(x_new_raw),
        nan=0.0,
        posinf=1e4,
        neginf=-1e4,
    ).astype(np.float32)

    return {
        "x_new_raw": x_new_raw,
        "metadata_df": metadata_df,
        "y_true": y_true,
        "sample_name_source": sample_name_source,
    }


def _ensure_output_dirs(results_dir, model_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)


def _build_run_prefix(args):
    if args.save_name:
        save_name = args.save_name
        if save_name.endswith(".pkl"):
            save_name = save_name[:-4]
        return save_name

    if args.new_data_npy:
        mode_tag = "SeqTransductive"
        pca_tag = f"_PCA{args.pca_n}D"
    elif args.inductive_embedding_mode == "raw" or args.pca_n <= 0:
        mode_tag = "Inductive"
        pca_tag = "_EmbRaw"
    else:
        mode_tag = "Inductive"
        pca_tag = f"_PCA{args.pca_n}D"

    return (
        f"lnp_1by1-{args.split_method}-{args.model_type}"
        f"-Seed{args.seed}_{args.embedding_name}_{mode_tag}{pca_tag}_eval{args.eval_split}"
    )


def build_ood_split(args):
    """Unified OOD split entry point, guaranteeing three splits (X_train, X_val, X_test, y_train, y_val, y_test)."""
    figure_path = Path(args.results_dir) / f"{_build_run_prefix(args)}_split.png"

    if args.split_method == "butina":
        from PCA_butina_split import get_butina_split_data

        return get_butina_split_data(
            data_csv=args.data_csv,
            embedding_npy=args.embedding_train,
            target_column=TARGET_COLUMN,
            target_classes=TARGET_CLASSES,
            radius=args.butina_radius,
            n_bits=args.butina_n_bits,
            cutoff=args.butina_cutoff,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            min_train_ratio=args.min_train_ratio,
            seed=args.seed,
            visualize=args.split_visualize,
            figure_path=figure_path,
            projection_metric=args.projection_metric,
            morgan_fps_path=getattr(args, "morgan_fps_path", None),
            verbose=False,
        )

    raise ValueError(f"[ERROR] Unsupported split_method: {args.split_method}")


def prepare_pca_features_from_arrays(X_train_raw, X_test_raw=None, X_new_raw=None, pca_n=20, seed=42):
    """Unified PCA feature preparation logic, supporting a single new sample participating in PCA fitting."""
    if X_train_raw.ndim != 2:
        raise ValueError("[ERROR] Training set features must be a 2D array.")
    if X_train_raw.shape[1] <= RATIO_COLS:
        raise ValueError(
            f"[ERROR] Feature dimension too small, cannot split last {RATIO_COLS} dimensions as ratio features."
        )

    total_dim = X_train_raw.shape[1]
    emb_dim = total_dim - RATIO_COLS

    def _split_blocks(x_raw, array_name):
        if x_raw is None:
            return None, None
        if x_raw.ndim != 2:
            raise ValueError(f"[ERROR] {array_name} must be a 2D array.")
        if x_raw.shape[1] != total_dim:
            raise ValueError(
                f"[ERROR] Feature dimension of {array_name} does not match training data. {array_name}={x_raw.shape[1]}, train={total_dim}"
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
        return x_emb, x_rat

    X_train_emb, X_train_rat = _split_blocks(X_train_raw, "X_train_raw")
    X_test_emb, X_test_rat = _split_blocks(X_test_raw, "X_test_raw")
    X_new_emb, X_new_rat = _split_blocks(X_new_raw, "X_new_raw")

    scaler = StandardScaler()
    X_train_emb_scaled = scaler.fit_transform(X_train_emb)
    X_test_emb_scaled = scaler.transform(X_test_emb) if X_test_emb is not None else None
    X_new_emb_scaled = scaler.transform(X_new_emb) if X_new_emb is not None else None

    X_for_pca_fit = X_train_emb_scaled
    if X_new_emb_scaled is not None:
        X_for_pca_fit = np.vstack([X_train_emb_scaled, X_new_emb_scaled])

    n_comp_actual = min(int(pca_n), X_for_pca_fit.shape[0], X_for_pca_fit.shape[1])
    if n_comp_actual < 1:
        raise ValueError("[ERROR] Effective PCA components less than 1, cannot continue training.")

    pca = PCA(n_components=n_comp_actual, random_state=seed)
    pca.fit(X_for_pca_fit)

    X_train_pca = pca.transform(X_train_emb_scaled)
    X_test_pca = pca.transform(X_test_emb_scaled) if X_test_emb_scaled is not None else None
    X_new_pca = pca.transform(X_new_emb_scaled) if X_new_emb_scaled is not None else None

    X_train_final = np.hstack([X_train_pca, X_train_rat]).astype(np.float32)
    X_test_final = (
        np.hstack([X_test_pca, X_test_rat]).astype(np.float32)
        if X_test_pca is not None
        else None
    )
    X_new_final = (
        np.hstack([X_new_pca, X_new_rat]).astype(np.float32)
        if X_new_pca is not None
        else None
    )

    meta = {
        "emb_dim": emb_dim,
        "ratio_cols": RATIO_COLS,
        "n_components_requested": int(pca_n),
        "n_components_actual": int(n_comp_actual),
        "used_new_data_for_pca": X_new_emb_scaled is not None,
        "new_data_count_for_pca": int(0 if X_new_emb_scaled is None else len(X_new_emb_scaled)),
    }
    return X_train_final, X_test_final, X_new_final, scaler, pca, meta


def prepare_transductive_pca_features(
    X_train_raw, X_val_raw, X_test_raw, pca_n, seed, inductive_embedding_mode="pca"
):
    """Extract features uniformly across all three splits: fitting strictly based on training set."""
    total_dim = X_train_raw.shape[1]
    emb_dim = total_dim - RATIO_COLS

    def _split_blocks(x_raw):
        if x_raw is None:
            return None, None
        x_emb = np.nan_to_num(x_raw[:, :emb_dim], nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        x_rat = np.nan_to_num(x_raw[:, emb_dim:], nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        return x_emb, x_rat

    X_train_emb, X_train_rat = _split_blocks(X_train_raw)
    X_val_emb, X_val_rat = _split_blocks(X_val_raw)
    X_test_emb, X_test_rat = _split_blocks(X_test_raw)

    if inductive_embedding_mode == "raw" or int(pca_n) <= 0:
        X_train_final = np.hstack([X_train_emb, X_train_rat]).astype(np.float32)
        X_val_final = np.hstack([X_val_emb, X_val_rat]).astype(np.float32) if X_val_emb is not None else None
        X_test_final = np.hstack([X_test_emb, X_test_rat]).astype(np.float32) if X_test_emb is not None else None
        meta = {
            "emb_dim": emb_dim,
            "ratio_cols": RATIO_COLS,
            "n_components_requested": None,
            "n_components_actual": None,
            "feature_mode": "raw",
        }
        return X_train_final, X_val_final, X_test_final, None, None, meta

    scaler = StandardScaler()
    X_train_emb_scaled = scaler.fit_transform(X_train_emb)
    X_val_emb_scaled = scaler.transform(X_val_emb) if X_val_emb is not None else None
    X_test_emb_scaled = scaler.transform(X_test_emb) if X_test_emb is not None else None

    n_comp_actual = min(int(pca_n), X_train_emb_scaled.shape[0], X_train_emb_scaled.shape[1])
    pca = PCA(n_components=n_comp_actual, random_state=seed)
    pca.fit(X_train_emb_scaled)

    X_train_pca = pca.transform(X_train_emb_scaled)
    X_val_pca = pca.transform(X_val_emb_scaled) if X_val_emb_scaled is not None else None
    X_test_pca = pca.transform(X_test_emb_scaled) if X_test_emb_scaled is not None else None

    X_train_final = np.hstack([X_train_pca, X_train_rat]).astype(np.float32)
    X_val_final = np.hstack([X_val_pca, X_val_rat]).astype(np.float32) if X_val_pca is not None else None
    X_test_final = np.hstack([X_test_pca, X_test_rat]).astype(np.float32) if X_test_pca is not None else None

    meta = {
        "emb_dim": emb_dim,
        "ratio_cols": RATIO_COLS,
        "n_components_requested": int(pca_n),
        "n_components_actual": int(n_comp_actual),
    }
    return X_train_final, X_val_final, X_test_final, scaler, pca, meta


def build_classifier(args):
    """Build classifier based on model_type."""
    model_key = args.model_type.lower()

    if model_key == "tabpfn":
        device = args.tabpfn_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return TabPFNClassifierWrapper(
            device=device,
            n_estimators=args.tabpfn_n_estimators,
            max_time=args.tabpfn_max_time,
        )

    if model_key in ["random_forest", "rf"]:
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if model_key in ["lightgbm", "lgbm"]:
        if LGBMClassifier is None:
            raise ImportError("[ERROR] lightgbm is not installed. Please run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=getattr(args, "lgbm_learning_rate", 0.05),
            num_leaves=getattr(args, "lgbm_num_leaves", 31),
            max_depth=getattr(args, "lgbm_max_depth", -1),
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )

    if model_key in ["svm", "svc"]:
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=args.svm_c,
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=args.seed,
            ),
        )

    if model_key == "transformer":
        return SklearnTransformer(
            epochs=args.trans_epochs,
            lr=args.trans_lr,
            device=args.trans_device,
            patience=args.trans_patience,
        )

    raise ValueError(f"[ERROR] Unsupported model_type: {args.model_type}")


def get_model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "steps") and len(model.steps) > 0:
        last_estimator = model.steps[-1][1]
        if hasattr(last_estimator, "classes_"):
            return list(last_estimator.classes_)
    return TARGET_CLASSES


def align_probabilities(raw_probabilities, model_classes, target_classes=TARGET_CLASSES):
    class_to_idx = {str(label): idx for idx, label in enumerate(model_classes)}
    aligned = np.zeros((raw_probabilities.shape[0], len(target_classes)), dtype=np.float32)
    for target_idx, target_label in enumerate(target_classes):
        if str(target_label) in class_to_idx:
            aligned[:, target_idx] = raw_probabilities[:, class_to_idx[str(target_label)]]
    return aligned


def predict_with_probabilities(model, X_input, target_classes=TARGET_CLASSES):
    y_pred = model.predict(X_input)
    if hasattr(model, "predict_proba"):
        raw_probabilities = model.predict_proba(X_input)
        probabilities = align_probabilities(
            raw_probabilities=raw_probabilities,
            model_classes=get_model_classes(model),
            target_classes=target_classes,
        )
    else:
        probabilities = np.zeros((len(y_pred), len(target_classes)), dtype=np.float32)
    return y_pred, probabilities


def build_prediction_dataframe(y_pred, probabilities, target_classes, y_true=None, metadata_df=None):
    if metadata_df is not None:
        pred_df = metadata_df.reset_index(drop=True).copy()
    else:
        pred_df = pd.DataFrame({"sample_index": np.arange(len(y_pred), dtype=int)})

    if y_true is not None:
        pred_df["y_true"] = np.asarray(y_true)
    pred_df["y_pred"] = np.asarray(y_pred)
    pred_df["pred_confidence"] = probabilities.max(axis=1)

    for class_idx, class_name in enumerate(target_classes):
        pred_df[f"prob_{class_name}"] = probabilities[:, class_idx]

    return pred_df


def _build_label_metrics(y_true, y_pred, label_space):
    report = classification_report(
        y_true,
        y_pred,
        labels=label_space,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_space)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": list(label_space),
    }


def evaluate_model(model, X_eval, y_eval):
    y_pred, probabilities = predict_with_probabilities(model, X_eval, TARGET_CLASSES)
    metrics = _build_label_metrics(y_eval, y_pred, TARGET_CLASSES)
    metrics["sample_count"] = int(len(y_eval))
    return y_pred, probabilities, metrics


def save_artifacts(args, run_prefix, save_package, metrics, prediction_df, eval_name, split_counts):
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)

    model_path = model_dir / f"{run_prefix}.pkl"
    metrics_path = results_dir / f"{run_prefix}_eval_{args.eval_split}_metrics.json"
    pred_path = results_dir / f"{run_prefix}_{args.eval_split}_predictions.csv"

    if not args.no_save_model:
        joblib.dump(save_package, model_path)
        print(f"[INFO] Model saved to: {model_path}")
    else:
        print(f"[INFO] [--no_save_model enabled] Skipping model weights saving: {model_path}")

    metrics_payload = {
        "run_name": run_prefix,
        "split_method": args.split_method,
        "model_type": args.model_type,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "eval_split_name": eval_name,
        "split_counts": split_counts,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

    prediction_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Evaluation metrics saved to: {metrics_path}")
    print(f"[INFO] {eval_name} predictions saved to: {pred_path}")

    return model_path if not args.no_save_model else None, metrics_path, pred_path


def save_transductive_prediction_results(args, run_prefix, metrics, prediction_df, pca_meta, split_counts):
    results_dir = Path(args.results_dir)
    metrics_path = results_dir / f"{run_prefix}_metrics.json"
    pred_path = results_dir / f"{run_prefix}_new_data_predictions.csv"

    metrics_payload = {
        "run_name": run_prefix,
        "split_method": args.split_method,
        "model_type": args.model_type,
        "seed": args.seed,
        "evaluation_target": "new_data_sequential_transductive",
        "prediction_csv": str(pred_path),
        "new_data_npy": args.new_data_npy,
        "new_data_csv": args.new_data_csv,
        "split_counts": split_counts,
        "pca_meta": pca_meta,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

    prediction_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Sequential transductive evaluation metrics saved to: {metrics_path}")
    print(f"[INFO] New sample predictions saved to: {pred_path}")

    return metrics_path, pred_path


def run_sequential_transductive_prediction(args, run_prefix, X_train_raw, y_train, split_counts):
    """
    Sequential transductive prediction:
    For each new sample, incorporate it into the train set PCA fitting stage, train a model and predict on it,
    then summarize all new sample predictions into a CSV and compute accuracy if ground truth labels exist.
    """
    bundle = load_new_data_bundle(
        new_data_npy=args.new_data_npy,
        new_data_csv=args.new_data_csv,
        target_column=TARGET_COLUMN,
    )
    x_new_raw = bundle["x_new_raw"]
    metadata_df = bundle["metadata_df"]
    y_true = bundle["y_true"]

    print(f"\n[INFO] Starting sequential transductive prediction, sample count: {len(x_new_raw)}")

    prediction_frames = []
    fit_times = []
    pca_meta_list = []

    for sample_idx in range(len(x_new_raw)):
        print(f"\n[Transductive 1-by-1] Sample {sample_idx + 1}/{len(x_new_raw)}")
        x_new_single = x_new_raw[sample_idx : sample_idx + 1]

        X_train_final, _, X_new_final, _, _, pca_meta = prepare_pca_features_from_arrays(
            X_train_raw=X_train_raw,
            X_test_raw=None,
            X_new_raw=x_new_single,
            pca_n=args.pca_n,
            seed=args.seed,
        )
        pca_meta_list.append(pca_meta)

        clf = build_classifier(args)
        fit_start = time.perf_counter()
        clf.fit(X_train_final, y_train)
        fit_time_seconds = float(time.perf_counter() - fit_start)
        fit_times.append(fit_time_seconds)

        y_pred_single, probabilities_single = predict_with_probabilities(
            clf,
            X_new_final,
            TARGET_CLASSES,
        )

        row_metadata = metadata_df.iloc[[sample_idx]].copy().reset_index(drop=True)
        row_metadata["transductive_order"] = int(sample_idx + 1)
        row_metadata["fit_time_seconds"] = fit_time_seconds
        row_y_true = None if y_true is None else y_true[sample_idx : sample_idx + 1]
        row_prediction = build_prediction_dataframe(
            y_pred=y_pred_single,
            probabilities=probabilities_single,
            target_classes=TARGET_CLASSES,
            y_true=row_y_true,
            metadata_df=row_metadata,
        )
        prediction_frames.append(row_prediction)

    prediction_df = pd.concat(prediction_frames, ignore_index=True)

    metrics = {
        "accuracy": None,
        "classification_report": None,
        "confusion_matrix": None,
        "labels": TARGET_CLASSES,
        "sample_count": int(len(prediction_df)),
        "labeled_sample_count": 0,
        "fit_time_seconds_total": float(sum(fit_times)),
        "fit_time_seconds_mean": float(np.mean(fit_times)) if fit_times else 0.0,
        "per_sample_pca": True,
    }

    if "y_true" in prediction_df.columns:
        valid_mask = prediction_df["y_true"].isin(TARGET_CLASSES).to_numpy()
        metrics["labeled_sample_count"] = int(valid_mask.sum())
        if valid_mask.sum() > 0:
            valid_y_true = prediction_df.loc[valid_mask, "y_true"].to_numpy()
            valid_y_pred = prediction_df.loc[valid_mask, "y_pred"].to_numpy()
            metrics.update(_build_label_metrics(valid_y_true, valid_y_pred, TARGET_CLASSES))

    if metrics["accuracy"] is None:
        print("[INFO] No valid Organ labels in new_data_csv, skipped accuracy computation.")
    else:
        print(f"[INFO] New sample transductive accuracy: {metrics['accuracy']:.4f}")

    pca_meta_summary = {
        "emb_dim": pca_meta_list[0]["emb_dim"] if pca_meta_list else None,
        "ratio_cols": pca_meta_list[0]["ratio_cols"] if pca_meta_list else None,
        "n_components_requested": pca_meta_list[0]["n_components_requested"] if pca_meta_list else int(args.pca_n),
        "n_components_actual": pca_meta_list[0]["n_components_actual"] if pca_meta_list else None,
        "used_new_data_for_pca": True,
        "new_data_count_for_prediction": int(len(x_new_raw)),
        "per_sample_pca": True,
    }

    save_transductive_prediction_results(
        args=args,
        run_prefix=run_prefix,
        metrics=metrics,
        prediction_df=prediction_df,
        pca_meta=pca_meta_summary,
        split_counts=split_counts,
    )

    return metrics["accuracy"]


def train_pipeline(args):
    _ensure_output_dirs(args.results_dir, args.model_dir)

    # Normalize model_type naming
    model_key = args.model_type.lower()
    if model_key in ["lightgbm", "lgbm"]:
        args.model_type = "lightgbm"
    elif model_key in ["rf", "random_forest"]:
        args.model_type = "rf"
    elif model_key in ["svc", "svm"]:
        args.model_type = "svc"
    elif model_key == "transformer":
        args.model_type = "transformer"

    run_prefix = _build_run_prefix(args)

    print("\n" + "=" * 80)
    print(f"[START] Training task: {run_prefix}")
    print("=" * 80)
    print(f"Data split method:       {args.split_method}")
    print(f"Classifier model:        {args.model_type}")
    print(f"Random seed:             {args.seed}")
    print(f"Target evaluation split: {args.eval_split.upper()} (specified via --eval_split {args.eval_split})")

    # 1. Perform OOD split to obtain three subsets (Train, Val, Test)
    split_res = build_ood_split(args)
    if len(split_res) == 6:
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = split_res
    else:
        raise ValueError("[ERROR] OOD split must return three complete subsets (Train, Val, Test).")

    total_samples = len(X_train_raw) + len(X_val_raw) + len(X_test_raw)
    train_count = len(X_train_raw)
    val_count = len(X_val_raw)
    test_count = len(X_test_raw)

    split_counts = {
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "total": total_samples,
        "train_pct": round(train_count / total_samples * 100, 2),
        "val_pct": round(val_count / total_samples * 100, 2),
        "test_pct": round(test_count / total_samples * 100, 2),
    }

    eval_label = "Validation set (Val)" if args.eval_split == "val" else "Test set (Test)"

    print("\n" + "=" * 80)
    print("[INFO] Dataset 3-Way Split Completed")
    print(f"  * Train set samples: {train_count:3d} ({split_counts['train_pct']:5.2f}%)  [Strictly used for feature fitting and model training, never evaluated for accuracy]")
    print(f"  * Val set samples  : {val_count:3d} ({split_counts['val_pct']:5.2f}%)")
    print(f"  * Test set samples : {test_count:3d} ({split_counts['test_pct']:5.2f}%)")
    print(f"  * Total samples    : {total_samples:3d} (100.0%)")
    print("-" * 80)
    print(f"  [*] Specified evaluation target: {eval_label} (specified via --eval_split {args.eval_split})")
    print("=" * 80 + "\n")

    # 2. If new_data_npy is provided, execute sequential transductive loop (1-by-1 PCA + prediction)
    if args.new_data_npy:
        return run_sequential_transductive_prediction(
            args=args,
            run_prefix=run_prefix,
            X_train_raw=X_train_raw,
            y_train=y_train,
            split_counts=split_counts,
        )

    feature_desc = "Raw Embedding (No PCA)" if (args.inductive_embedding_mode == "raw" or args.pca_n <= 0) else f"PCA={args.pca_n}"
    print(f"[INFO] Starting feature processing ({feature_desc}) ...")
    X_train_final, X_val_final, X_test_final, scaler, pca, pca_meta = prepare_transductive_pca_features(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        pca_n=args.pca_n,
        seed=args.seed,
        inductive_embedding_mode=args.inductive_embedding_mode,
    )
    print(f"[INFO] Feature processing completed: Train shape={X_train_final.shape}, Val shape={X_val_final.shape}, Test shape={X_test_final.shape}")

    # 3. Train model (strictly on training set)
    print(f"\n[MODEL] Training classifier: {args.model_type} ...")
    clf = build_classifier(args)
    clf.fit(X_train_final, y_train)

    # 4. Evaluate (never evaluate on training set; evaluate Val or Test depending on eval_split)
    if args.eval_split == "val":
        X_eval, y_eval = X_val_final, y_val
    else:
        X_eval, y_eval = X_test_final, y_test

    y_pred, probabilities, metrics = evaluate_model(clf, X_eval, y_eval)
    print("\n" + "=" * 80)
    print(f"[STATS] Model Evaluation Results")
    print(f"  * Evaluation split : {eval_label}")
    print(f"  * Sample count     : {len(y_eval)}")
    print(f"  * Accuracy         : {metrics['accuracy']:.4f}")
    print("=" * 80)

    prediction_df = build_prediction_dataframe(
        y_pred=y_pred,
        probabilities=probabilities,
        target_classes=TARGET_CLASSES,
        y_true=y_eval,
        metadata_df=None,
    )

    # 5. Save model and metrics
    save_package = {
        "scaler": scaler,
        "pca": pca,
        "model": clf,
        "target_classes": TARGET_CLASSES,
        "target_column": TARGET_COLUMN,
        "seed": args.seed,
        "split_method": args.split_method,
        "model_type": args.model_type,
        "eval_split": args.eval_split,
        "pca_meta": pca_meta,
        "split_counts": split_counts,
    }
    save_artifacts(
        args=args,
        run_prefix=run_prefix,
        save_package=save_package,
        metrics=metrics,
        prediction_df=prediction_df,
        eval_name=eval_label,
        split_counts=split_counts,
    )

    return metrics["accuracy"]


def main():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[INFO] GPU ready: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass

    args = parse_arguments()
    train_pipeline(args)


if __name__ == "__main__":
    main()
