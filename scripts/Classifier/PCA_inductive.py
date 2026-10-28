# Inductive Learning: Cluster OOD Split & Model Evaluation Pipeline
# Strictly Inductive: StandardScaler and PCA are fitted exclusively on the training set (X_train).
# Zero data leakage: Out-of-distribution validation (val) and test (test) sets are strictly transformed out-of-sample.

import argparse
import copy
import json
import os
import sys
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


# ================= TabPFN Wrapper =================
class TabPFNClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, device="cuda", n_estimators=8):
        self.device = device
        self.n_estimators = n_estimators
        self.model_ = None
        self.classes_ = None

    def _resolve_device(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] GPU unavailable, TabPFN falling back to CPU.")
            return "cpu"
        return self.device

    def fit(self, X, y):
        from tabpfn import TabPFNClassifier

        device = self._resolve_device()
        try:
            # Compatible with newer TabPFN (v2.x / v6.x, using n_estimators, random_state)
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
                # Compatible with older TabPFN (v0.1.x, using N_ensemble_configurations, only_inference)
                self.model_ = TabPFNClassifier(
                    device=device,
                    N_ensemble_configurations=self.n_estimators,
                    only_inference=True,
                    seed=42,
                )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
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
        description="Inductive LNP model training and evaluation script based on cluster OOD split (supports controllable val/test set evaluation)"
    )

    # Data and paths
    parser.add_argument("--data_csv", type=str, required=True, help="Path to LNP formulation data CSV")
    parser.add_argument(
        "--embedding_train",
        type=str,
        required=True,
        help="Path to .npy file containing embeddings for all 447 samples",
    )
    parser.add_argument("--results_dir", type=str, required=True, help="Directory to save evaluation results")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save trained models")

    # Run naming
    parser.add_argument("--embedding_name", type=str, default="Embedding", help="Embedding name used for output naming (default: Embedding)")
    parser.add_argument("--save_name", type=str, default="", help="Custom model save filename prefix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Core evaluation target selection
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
    parser.add_argument("--pca_n", type=int, default=20, help="PCA dimension reduction target (<=0 disables PCA, default: 20)")
    parser.add_argument(
        "--inductive_embedding_mode",
        "--embedding_mode",
        type=str,
        default="pca",
        choices=["pca", "raw"],
        dest="inductive_embedding_mode",
        help="Embedding mode: 'pca' (perform PCA dimensionality reduction) or 'raw' (use original embedding) (default: pca)",
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        default="tabpfn",
        choices=[
            "tabpfn",
            "lightgbm",
            "lgbm",
            "LightGBM",
            "rf",
            "random_forest",
            "svc",
            "svm",
            "transformer",
        ],
        help="Classifier type (default: tabpfn)",
    )

    # TabPFN parameters
    parser.add_argument("--tabpfn_device", type=str, default="cuda", help="TabPFN device (default: cuda)")
    parser.add_argument("--tabpfn_n_estimators", type=int, default=32, help="TabPFN number of ensemble estimators (default: 32)")

    # LightGBM parameters
    parser.add_argument("--lgbm_n_estimators", type=int, default=100, help="LightGBM n_estimators (default: 100)")
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05, help="LightGBM learning rate (default: 0.05)")
    parser.add_argument("--lgbm_num_leaves", type=int, default=30, help="LightGBM num_leaves (default: 30)")
    parser.add_argument("--lgbm_max_depth", type=int, default=-1, help="LightGBM max_depth (default: -1)")

    # Random Forest parameters
    parser.add_argument("--rf_n_estimators", type=int, default=100, help="Random Forest n_estimators (default: 100)")
    parser.add_argument("--rf_max_depth", type=int, default=None, help="Random Forest max_depth (default: None)")

    # SVC parameters
    parser.add_argument("--svc_c", type=float, default=1.0, help="SVC regularization parameter C (default: 1.0)")
    parser.add_argument("--svc_kernel", type=str, default="rbf", help="SVC kernel (default: rbf)")

    # Transformer parameters
    parser.add_argument("--transformer_epochs", type=int, default=200, help="Transformer training epochs (default: 200)")
    parser.add_argument("--transformer_lr", type=float, default=0.0005, help="Transformer learning rate (default: 0.0005)")
    parser.add_argument("--transformer_device", type=str, default="cuda:0", help="Transformer device (default: cuda:0)")
    parser.add_argument("--transformer_patience", type=int, default=15, help="Transformer early stopping patience (default: 15)")

    # Storage control parameters
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="Do not save trained model weights .pkl file to save disk space",
    )

    return parser.parse_args()


def _ensure_output_dirs(results_dir, model_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)


def _build_run_prefix(args):
    if args.save_name:
        save_name = args.save_name
        if save_name.endswith(".pkl"):
            save_name = save_name[:-4]
        return save_name

    if args.inductive_embedding_mode == "raw" or args.pca_n <= 0:
        embedding_variant = "EmbRaw"
    else:
        embedding_variant = f"EmbPCA{args.pca_n}D"

    return (
        f"lnp_cluster-{args.split_method}-{args.model_type}"
        f"-Seed{args.seed}_{args.embedding_name}_{embedding_variant}_eval{args.eval_split}"
    )


def build_ood_split(args):
    """
    Unified OOD split entry point, guaranteeing three splits: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
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


def prepare_model_features(
    X_train_raw,
    X_val_raw,
    X_test_raw,
    pca_n=20,
    seed=42,
    inductive_embedding_mode="pca",
):
    """
    Unified feature engineering across three subsets (strictly inductive):
    - StandardScaler and PCA are fitted strictly on X_train only;
    - Strict out-of-sample transform is applied to X_val and X_test;
    - Zero data leakage.
    """
    total_cols = X_train_raw.shape[1]
    emb_dim = total_cols - RATIO_COLS
    if emb_dim <= 0:
        raise ValueError(f"[ERROR] Total feature columns ({total_cols}) must be greater than ratio columns ({RATIO_COLS}).")

    X_train_emb = X_train_raw[:, :emb_dim]
    X_train_rat = X_train_raw[:, emb_dim:]

    X_val_emb = X_val_raw[:, :emb_dim] if X_val_raw is not None else None
    X_val_rat = X_val_raw[:, emb_dim:] if X_val_raw is not None else None

    X_test_emb = X_test_raw[:, :emb_dim]
    X_test_rat = X_test_raw[:, emb_dim:]

    use_pca = inductive_embedding_mode == "pca" and int(pca_n) > 0

    # Clean numerical values
    X_train_emb = np.nan_to_num(X_train_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test_emb = np.nan_to_num(X_test_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_train_rat = np.nan_to_num(X_train_rat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    X_test_rat = np.nan_to_num(X_test_rat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if X_val_emb is not None:
        X_val_emb = np.nan_to_num(X_val_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
        X_val_rat = np.nan_to_num(X_val_rat, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if not use_pca:
        X_train_final = np.hstack([X_train_emb, X_train_rat]).astype(np.float32)
        X_val_final = np.hstack([X_val_emb, X_val_rat]).astype(np.float32) if X_val_emb is not None else None
        X_test_final = np.hstack([X_test_emb, X_test_rat]).astype(np.float32)
        meta = {
            "emb_dim": emb_dim,
            "ratio_cols": RATIO_COLS,
            "n_components_requested": None,
            "n_components_actual": None,
            "learning_mode": "inductive",
            "feature_mode": "raw",
        }
        return X_train_final, X_val_final, X_test_final, None, None, meta

    # Standard scaling fitted strictly on training set
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb = scaler.transform(X_test_emb)
    if X_val_emb is not None:
        X_val_emb = scaler.transform(X_val_emb)

    n_comp_actual = min(int(pca_n), X_train_emb.shape[0], X_train_emb.shape[1])
    if n_comp_actual < 1:
        raise ValueError("[ERROR] Effective PCA components less than 1, cannot continue training.")

    pca = PCA(n_components=n_comp_actual, random_state=seed)
    pca.fit(X_train_emb)

    X_train_pca = pca.transform(X_train_emb)
    X_test_pca = pca.transform(X_test_emb)
    X_val_pca = pca.transform(X_val_emb) if X_val_emb is not None else None

    X_train_final = np.hstack([X_train_pca, X_train_rat]).astype(np.float32)
    X_val_final = np.hstack([X_val_pca, X_val_rat]).astype(np.float32) if X_val_pca is not None else None
    X_test_final = np.hstack([X_test_pca, X_test_rat]).astype(np.float32)

    meta = {
        "emb_dim": emb_dim,
        "ratio_cols": RATIO_COLS,
        "n_components_requested": int(pca_n),
        "n_components_actual": int(n_comp_actual),
        "learning_mode": "inductive",
        "feature_mode": "pca",
    }
    return X_train_final, X_val_final, X_test_final, scaler, pca, meta


def build_classifier(args):
    """Build classifier based on model_type."""
    model_key = args.model_type.lower()
    if model_key == "tabpfn":
        return TabPFNClassifierWrapper(
            device=args.tabpfn_device,
            n_estimators=args.tabpfn_n_estimators,
        )

    if model_key in ["lightgbm", "lgbm"]:
        if LGBMClassifier is None:
            raise ImportError("[ERROR] lightgbm is not installed. Please run: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=args.lgbm_n_estimators,
            learning_rate=args.lgbm_learning_rate,
            num_leaves=args.lgbm_num_leaves,
            max_depth=args.lgbm_max_depth,
            random_state=args.seed,
            n_jobs=-1,
            verbose=-1,
        )

    if model_key in ["rf", "random_forest"]:
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
            n_jobs=-1,
        )

    if model_key in ["svc", "svm"]:
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=args.svc_c,
                kernel=args.svc_kernel,
                probability=True,
                random_state=args.seed,
            ),
        )

    if model_key == "transformer":
        return SklearnTransformer(
            epochs=args.transformer_epochs,
            lr=args.transformer_lr,
            device=args.transformer_device,
            patience=args.transformer_patience,
        )

    raise ValueError(f"[ERROR] Unsupported model_type: {args.model_type}")


def evaluate_model(model, X_eval, y_eval):
    """Evaluate model performance on specified subset."""
    y_pred = model.predict(X_eval)
    acc = accuracy_score(y_eval, y_pred)
    report = classification_report(
        y_eval,
        y_pred,
        labels=TARGET_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_eval, y_pred, labels=TARGET_CLASSES)

    metrics = {
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": TARGET_CLASSES,
    }
    return y_pred, metrics


def save_artifacts(
    args,
    run_prefix,
    save_package,
    metrics,
    y_eval,
    y_pred,
    eval_name,
    split_counts,
):
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

    feature_mode = "pca" if (args.inductive_embedding_mode == "pca" and args.pca_n > 0) else "raw"
    metrics_payload = {
        "run_name": run_prefix,
        "split_method": args.split_method,
        "model_type": args.model_type,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "eval_split_name": eval_name,
        "learning_mode": "inductive",
        "feature_mode": feature_mode,
        "split_counts": split_counts,
        "val_ratio_target": args.val_ratio,
        "test_ratio_target": args.test_ratio,
        "min_train_ratio": args.min_train_ratio,
        "metrics": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as fp:
        json.dump(metrics_payload, fp, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({"y_true": y_eval, "y_pred": y_pred})
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] Evaluation metrics saved to: {metrics_path}")
    print(f"[INFO] {eval_name} predictions saved to: {pred_path}")

    return model_path if not args.no_save_model else None, metrics_path, pred_path


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
    print(f"Learning paradigm:       INDUCTIVE")
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

    # Print dataset split statistics before model training
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

    # 2. Feature engineering pipeline (strictly fitted on Train only)
    feature_mode = "pca" if (args.inductive_embedding_mode == "pca" and args.pca_n > 0) else "raw"
    feature_desc = "Raw Embedding (No PCA)" if (feature_mode == "raw") else f"PCA={args.pca_n}"

    print(f"[INFO] Starting inductive feature processing (feature_mode={feature_mode}, PCA={args.pca_n}) ...")
    X_train_final, X_val_final, X_test_final, scaler, pca, pca_meta = prepare_model_features(
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        pca_n=args.pca_n,
        seed=args.seed,
        inductive_embedding_mode=args.inductive_embedding_mode,
    )
    print(f"[INFO] Feature processing completed: Train shape={X_train_final.shape}, Val shape={X_val_final.shape}, Test shape={X_test_final.shape}")
    if pca_meta["feature_mode"] == "pca":
        print(f"[INFO] Actual PCA components: {pca_meta['n_components_actual']}")

    # 3. Train model (strictly using training set)
    print(f"\n[MODEL] Training classifier: {args.model_type} ...")
    clf = build_classifier(args)
    clf.fit(X_train_final, y_train)

    # 4. Evaluate accuracy (never evaluate on training set; evaluate Val or Test depending on eval_split)
    if args.eval_split == "val":
        X_eval, y_eval = X_val_final, y_val
    else:
        X_eval, y_eval = X_test_final, y_test

    y_pred, metrics = evaluate_model(clf, X_eval, y_eval)
    print("\n" + "=" * 80)
    print(f"[STATS] Model Evaluation Results")
    print(f"  * Evaluation split : {eval_label}")
    print(f"  * Sample count     : {len(y_eval)}")
    print(f"  * Accuracy         : {metrics['accuracy']:.4f}")
    print("=" * 80)

    # 5. Save model and evaluation metrics
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
        "split_params": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_train_ratio": args.min_train_ratio,
            "butina_radius": args.butina_radius,
            "butina_n_bits": args.butina_n_bits,
            "butina_cutoff": args.butina_cutoff,
        },
        "pca_meta": pca_meta,
        "embedding_name": args.embedding_name,
        "emb_dim": pca_meta["emb_dim"],
        "learning_mode": "inductive",
        "feature_mode": pca_meta["feature_mode"],
        "split_counts": split_counts,
    }
    save_artifacts(
        args=args,
        run_prefix=run_prefix,
        save_package=save_package,
        metrics=metrics,
        y_eval=y_eval,
        y_pred=y_pred,
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
