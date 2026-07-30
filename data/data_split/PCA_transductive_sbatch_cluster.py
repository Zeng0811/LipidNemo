import argparse
import json
import os
import sys
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# Force HuggingFace offline mode for reproducibility and security.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["AG_MAX_MEMORY_USAGE_RATIO"] = "100.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")

TARGET_COLUMN = "Organ"
TARGET_CLASSES = ["Liver", "Lung", "Spleen", "None"]
RATIO_COLS = 5


def print_torch_cuda_runtime_info():
    """Print CUDA/GPU runtime info detected by current Python process for troubleshooting."""
    print("CUDA Runtime Info:")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"  SLURM_JOB_GPUS={os.environ.get('SLURM_JOB_GPUS', '<unset>')}")
    print(f"  SLURM_GPUS_ON_NODE={os.environ.get('SLURM_GPUS_ON_NODE', '<unset>')}")
    print(f"  torch.__version__ = {torch.__version__}")
    print(f"  torch.version.cuda = {torch.version.cuda}")
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}")

    visible_count = torch.cuda.device_count()
    for gpu_idx in range(visible_count):
        try:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
        except Exception as exc:
            gpu_name = f"<Failed to read: {exc}>"
        print(f"  visible_gpu[{gpu_idx}] = {gpu_name}")

    if torch.cuda.is_available():
        try:
            current_idx = torch.cuda.current_device()
            current_name = torch.cuda.get_device_name(current_idx)
            print(f"  current_gpu[{current_idx}] = {current_name}")
        except Exception as exc:
            print(f"  current_gpu = <Failed to read: {exc}>")
    else:
        print("  current_gpu = <unavailable because torch.cuda.is_available() is False>")


def log_selected_torch_device(device, model_name):
    """Print actual hardware device used by model (distinguishing requested vs active device)."""
    device_str = str(device)

    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                f"[{model_name}] Actual device: cpu "
                f"(Requested={device_str}, Reason=torch.cuda.is_available()=False)"
            )
            return

        try:
            torch_device = torch.device(device_str)
            device_idx = torch_device.index
            if device_idx is None:
                device_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_idx)
            print(f"[{model_name}] Actual device: {device_str} -> GPU {device_idx}: {gpu_name}")
        except Exception as exc:
            print(f"[{model_name}] Actual device: {device_str} -> <Failed to read GPU name: {exc}>")
        return

    print(f"[{model_name}] Actual device: {device_str}")


# ================= Transformer (Scratch) =================
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
        x = x.squeeze(1)
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
            print("Warning: CUDA unavailable for Transformer, falling back to CPU.")
            return "cpu"
        return self.device

    def _build_model(self, resolved_device=None):
        if resolved_device is None:
            resolved_device = self._resolve_device()
        self.model = SimpleTransformerModule(
            input_dim=self.input_dim_,
            num_classes=self.num_classes_,
        ).to(resolved_device)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("Transformer input features must be a 2D array.")
        if len(X) < 2:
            raise ValueError("Transformer requires at least 2 samples.")

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        self.input_dim_ = X.shape[1] if self.input_dim is None else int(self.input_dim)
        if self.input_dim_ != X.shape[1]:
            raise ValueError(
                f"Transformer input_dim={self.input_dim_} does not match feature dimension {X.shape[1]}."
            )

        inferred_num_classes = len(self.classes_)
        if self.num_classes is not None and int(self.num_classes) < inferred_num_classes:
            raise ValueError(
                f"Transformer num_classes={self.num_classes} is smaller than inferred classes count {inferred_num_classes}."
            )
        self.num_classes_ = inferred_num_classes

        val_size = min(max(1, int(np.ceil(len(y_enc) * 0.1))), len(y_enc) - 1)
        stratify_y = y_enc if len(np.unique(y_enc)) > 1 and np.min(np.bincount(y_enc)) >= 2 else None
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_enc, test_size=val_size, random_state=42, stratify=stratify_y
            )
        except ValueError:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y_enc, test_size=val_size, random_state=42, stratify=None
            )

        device = self._resolve_device()
        self._build_model(device)
        log_selected_torch_device(device, "Transformer")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-2)

        train_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)

        best_val_loss = float('inf')
        best_model_weights = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(
                    f"      [Early Stopping] Triggered at epoch {epoch + 1}. "
                    f"Best weights loaded from epoch {epoch + 1 - self.patience}."
                )
                break

        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
        self.model.eval()

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(self._resolve_device())
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        predicted = predicted.cpu().numpy()
        if self.label_encoder_ is not None:
            return self.label_encoder_.inverse_transform(predicted)
        return predicted

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.asarray(X, dtype=np.float32)).to(self._resolve_device())
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        return probas.cpu().numpy()

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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LNP PCA + TabPFN/Baseline training script based on cluster OOD split"
    )

    # Data Inputs
    parser.add_argument("--data_csv", type=str, required=True, help="Path to label CSV file")
    parser.add_argument("--embedding_train", type=str, required=True, help="Path to training embedding .npy/.npz file")
    parser.add_argument("--results_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save trained models")
    parser.add_argument("--new_data_npy", type=str, default="", help="Path to new samples .npy for transductive PCA")

    # Run Naming
    parser.add_argument("--embedding_name", type=str, default="Embedding", help="Embedding name used for naming outputs")
    parser.add_argument("--save_name", type=str, default="", help="Custom model checkpoint filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Cluster Splitting
    parser.add_argument(
        "--split_method",
        type=str,
        default="hdbscan",
        choices=["butina", "hdbscan"],
        help="Dataset splitting method",
    )
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Target test set ratio")
    parser.add_argument("--split_visualize", action="store_true", help="Save cluster split visualization figure")

    # Butina Parameters
    parser.add_argument("--butina_radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--butina_n_bits", type=int, default=2048, help="Morgan fingerprint bit size")
    parser.add_argument(
        "--butina_cutoff",
        type=float,
        default=0.6,
        help="Butina distance cutoff; smaller value leads to stricter split",
    )

    # HDBSCAN Parameters
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=12, help="HDBSCAN min_cluster_size")
    parser.add_argument("--hdbscan_min_samples", type=int, default=None, help="HDBSCAN min_samples")
    parser.add_argument("--hdbscan_metric", type=str, default="euclidean", help="HDBSCAN distance metric")
    parser.add_argument(
        "--hdbscan_cluster_selection_epsilon",
        type=float,
        default=0.0,
        help="HDBSCAN cluster_selection_epsilon",
    )

    # PCA Parameters
    parser.add_argument("--pca_n", type=int, default=80, help="Target PCA reduced dimension count")

    # Model Parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="tabpfn",
        choices=[
            "tabpfn",
            "LightGBM",
            "random_forest",
            "logistic_regression",
            "Transformer",
            "svm",
        ],
        help="Model architecture to train",
    )

    # TabPFN Parameters
    parser.add_argument("--tabpfn_n_estimators", type=int, default=4, help="TabPFN ensemble size")
    parser.add_argument("--tabpfn_max_time", type=int, default=900, help="TabPFN max training time in seconds")
    parser.add_argument("--tabpfn_device", type=str, default="cuda:0", help="TabPFN compute device (e.g. cuda:0 / cpu)")

    # Baseline Parameters
    parser.add_argument("--rf_n_estimators", type=int, default=300, help="RandomForest estimator count")
    parser.add_argument("--lgbm_n_estimators", type=int, default=100, help="LightGBM estimator count")
    parser.add_argument("--svm_c", type=float, default=1.0, help="SVM C regularization parameter")
    parser.add_argument("--logreg_c", type=float, default=1.0, help="LogisticRegression C regularization parameter")
    parser.add_argument("--transformer_epochs", type=int, default=200, help="Transformer training epoch count")
    parser.add_argument("--transformer_lr", type=float, default=5e-4, help="Transformer learning rate")
    parser.add_argument("--transformer_device", type=str, default="cuda:0", help="Transformer device (e.g. cuda:0 / cpu)")
    parser.add_argument("--transformer_patience", type=int, default=15, help="Transformer early stopping patience")

    return parser.parse_args()


def data_load(file_path):
    """Smart data loader automatically supporting .npy / .npz / .csv formats."""
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
            print(f"Unsupported file format: {ext}")
            return None
        return data
    except Exception as exc:
        print(f"Read failure: {exc}")
        return None


def _ensure_output_dirs(results_dir, model_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)


def _build_run_prefix(args):
    if args.save_name:
        save_name = args.save_name
        if save_name.endswith(".pkl"):
            save_name = save_name[:-4]
        return save_name

    return (
        f"lnp_cluster-{args.split_method}-{args.model_type}"
        f"-Seed{args.seed}_{args.embedding_name}_{args.pca_n}D"
    )


def build_ood_split(args):
    """
    Unified OOD split entry point.

    Downstream pipeline relies only on X_train/X_test/y_train/y_test,
    so PCA, TabPFN, and baselines remain completely decoupled from underlying Butina or HDBSCAN logic.
    """
    figure_path = Path(args.results_dir) / f"{_build_run_prefix(args)}_split.png"

    if args.split_method == "butina":
        from PCA_transductive_butina_split import get_butina_split_data

        return get_butina_split_data(
            data_csv=args.data_csv,
            embedding_npy=args.embedding_train,
            target_column=TARGET_COLUMN,
            target_classes=TARGET_CLASSES,
            radius=args.butina_radius,
            n_bits=args.butina_n_bits,
            cutoff=args.butina_cutoff,
            test_ratio=args.test_ratio,
            seed=args.seed,
            visualize=args.split_visualize,
            figure_path=figure_path,
            verbose=True,
        )

    if args.split_method == "hdbscan":
        from PCA_transductive_hdbscan_split import get_hdbscan_split_data

        return get_hdbscan_split_data(
            data_csv=args.data_csv,
            embedding_npy=args.embedding_train,
            target_column=TARGET_COLUMN,
            target_classes=TARGET_CLASSES,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            metric=args.hdbscan_metric,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
            visualize=args.split_visualize,
            figure_path=figure_path,
            verbose=True,
        )

    raise ValueError(f"Unsupported split_method: {args.split_method}")


def prepare_transductive_pca_features(X_train_raw, X_test_raw, new_data_npy, pca_n, seed):
    """
    Transductive PCA feature pipeline:
    - Last 5 dimensions are preserved as formulation ratio features (bypassing PCA);
    - PCA is applied solely to embedding features;
    - If new_data_npy is provided, its embedding features are combined to fit PCA transductively.
    """
    if X_train_raw.ndim != 2 or X_test_raw.ndim != 2:
        raise ValueError("Train and test feature arrays must be 2D.")
    if X_train_raw.shape[1] <= RATIO_COLS:
        raise ValueError(
            f"Feature dimension too small to split last {RATIO_COLS} columns as ratio features."
        )

    emb_dim = X_train_raw.shape[1] - RATIO_COLS

    X_train_emb = X_train_raw[:, :emb_dim]
    X_test_emb = X_test_raw[:, :emb_dim]
    X_train_rat = X_train_raw[:, emb_dim:]
    X_test_rat = X_test_raw[:, emb_dim:]

    X_new_raw = None
    X_new_emb = None
    if new_data_npy:
        if not Path(new_data_npy).exists():
            print(f"Warning: new_data_npy not found, falling back to standard inductive PCA: {new_data_npy}")
        else:
            X_new_raw = data_load(new_data_npy)
            if X_new_raw is None:
                raise ValueError(f"Failed to load new_data_npy: {new_data_npy}")
            if X_new_raw.ndim != 2:
                raise ValueError("new_data_npy must be a 2D array.")
            if X_new_raw.shape[1] != X_train_raw.shape[1]:
                raise ValueError(
                    "Feature dimension mismatch between new_data_npy and train features. "
                    f"new={X_new_raw.shape[1]}, train={X_train_raw.shape[1]}"
                )
            X_new_emb = X_new_raw[:, :emb_dim]
            print(f"Loaded {X_new_raw.shape[0]} new samples for transductive PCA.")

    # Clean and standardize embedding portion only.
    X_train_emb = np.nan_to_num(X_train_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test_emb = np.nan_to_num(X_test_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)

    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb = scaler.transform(X_test_emb)

    if X_new_emb is not None:
        X_new_emb = np.nan_to_num(X_new_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        X_new_emb = scaler.transform(X_new_emb)
        X_for_pca_fit = np.vstack([X_train_emb, X_new_emb])
    else:
        X_for_pca_fit = X_train_emb

    n_comp_actual = min(int(pca_n), X_for_pca_fit.shape[0], X_for_pca_fit.shape[1])
    if n_comp_actual < 1:
        raise ValueError("Effective PCA dimension is less than 1, cannot continue.")

    pca = PCA(n_components=n_comp_actual, random_state=seed)
    pca.fit(X_for_pca_fit)

    X_train_pca = pca.transform(X_train_emb)
    X_test_pca = pca.transform(X_test_emb)

    # Concatenate ratio features back directly without PCA transformation.
    X_train_rat = np.nan_to_num(X_train_rat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    X_test_rat = np.nan_to_num(X_test_rat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    X_train_final = np.hstack([X_train_pca, X_train_rat])
    X_test_final = np.hstack([X_test_pca, X_test_rat])

    meta = {
        "emb_dim": emb_dim,
        "ratio_cols": RATIO_COLS,
        "n_components_requested": int(pca_n),
        "n_components_actual": int(n_comp_actual),
        "used_new_data_for_pca": X_new_emb is not None,
    }
    return X_train_final, X_test_final, scaler, pca, meta


def build_classifier(args):
    """Instantiate classifier based on model_type."""
    if args.model_type == "tabpfn":
        try:
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
                AutoTabPFNClassifier,
            )
        except ImportError as exc:
            raise ImportError(
                "tabpfn_extensions package is not installed, but model_type=tabpfn was specified."
            ) from exc

        device = args.tabpfn_device
        try:
            import torch

            if device.startswith("cuda") and not torch.cuda.is_available():
                print("Warning: CUDA is unavailable, TabPFN automatically falling back to CPU.")
                device = "cpu"
        except ImportError:
            if device.startswith("cuda"):
                print("Warning: PyTorch is not installed, TabPFN automatically falling back to CPU.")
                device = "cpu"

        log_selected_torch_device(device, "TabPFN")
        return AutoTabPFNClassifier(
            ignore_pretraining_limits=True,
            random_state=args.seed,
            n_estimators=args.tabpfn_n_estimators,
            max_time=args.tabpfn_max_time,
            device=device,
        )

    if args.model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            random_state=args.seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    if args.model_type == "LightGBM":
        return LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            min_child_samples=15,
            class_weight='balanced',
            random_state=args.seed, 
            verbose=-1
        )

    if args.model_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=args.logreg_c,
                random_state=args.seed,
                max_iter=2000,
                class_weight="balanced",
            ),
        )

    if args.model_type == "svm":
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

    if args.model_type == "Transformer":
        return SklearnTransformer(
            epochs=args.transformer_epochs,
            lr=args.transformer_lr,
            device=args.transformer_device,
            patience=args.transformer_patience,
        )

    raise ValueError(f"Unsupported model_type: {args.model_type}")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=TARGET_CLASSES)

    metrics = {
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": TARGET_CLASSES,
    }
    return y_pred, metrics


def save_artifacts(args, run_prefix, save_package, metrics, y_test, y_pred):
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)

    model_path = model_dir / f"{run_prefix}.pkl"
    metrics_path = results_dir / f"{run_prefix}_metrics.json"
    pred_path = results_dir / f"{run_prefix}_test_predictions.csv"

    joblib.dump(save_package, model_path)

    metrics_payload = {
        "run_name": run_prefix,
        "split_method": args.split_method,
        "model_type": args.model_type,
        "seed": args.seed,
        "test_ratio_target": args.test_ratio,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Test predictions saved: {pred_path}")

    return model_path, metrics_path, pred_path


def train_pipeline(args):
    _ensure_output_dirs(args.results_dir, args.model_dir)
    run_prefix = _build_run_prefix(args)

    print("\n" + "=" * 80)
    print(f"Starting training: {run_prefix}")
    print("=" * 80)
    print(f"Split method: {args.split_method}")
    print(f"Model type: {args.model_type}")
    print(f"Seed: {args.seed}")

    # 1. Obtain train/test splits through unified clustering interface (removing random split).
    X_train_raw, X_test_raw, y_train, y_test = build_ood_split(args)
    print(f"Raw train shape: {X_train_raw.shape}")
    print(f"Raw test shape: {X_test_raw.shape}")

    # 2. Perform transductive/inductive PCA on fixed OOD splits.
    pca_mode = "Transductive" if args.new_data_npy else "Inductive"
    print(f"\nExecuting {pca_mode} PCA (target n_components={args.pca_n}) ...")
    X_train_final, X_test_final, scaler, pca, pca_meta = prepare_transductive_pca_features(
        X_train_raw=X_train_raw,
        X_test_raw=X_test_raw,
        new_data_npy=args.new_data_npy,
        pca_n=args.pca_n,
        seed=args.seed,
    )
    print(f"PCA train shape: {X_train_final.shape}")
    print(f"PCA test shape: {X_test_final.shape}")
    print(f"Actual PCA n_components: {pca_meta['n_components_actual']}")

    # 3. Model training.
    print(f"\nTraining model: {args.model_type}")
    clf = build_classifier(args)
    clf.fit(X_train_final, y_train)

    # 4. Evaluation.
    y_pred, metrics = evaluate_model(clf, X_test_final, y_test)
    print(f"Test set accuracy: {metrics['accuracy']:.4f}")

    # 5. Save checkpoints and evaluation metrics.
    save_package = {
        "scaler": scaler,
        "pca": pca,
        "model": clf,
        "target_classes": TARGET_CLASSES,
        "target_column": TARGET_COLUMN,
        "seed": args.seed,
        "split_method": args.split_method,
        "model_type": args.model_type,
        "split_params": {
            "test_ratio": args.test_ratio,
            "butina_radius": args.butina_radius,
            "butina_n_bits": args.butina_n_bits,
            "butina_cutoff": args.butina_cutoff,
            "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
            "hdbscan_min_samples": args.hdbscan_min_samples,
            "hdbscan_metric": args.hdbscan_metric,
            "hdbscan_cluster_selection_epsilon": args.hdbscan_cluster_selection_epsilon,
        },
        "pca_meta": pca_meta,
        "embedding_name": args.embedding_name,
    }
    save_artifacts(args, run_prefix, save_package, metrics, y_test, y_pred)

    return metrics["accuracy"]


def main():
    print_torch_cuda_runtime_info()
    args = parse_arguments()
    train_pipeline(args)


if __name__ == "__main__":
    main()
