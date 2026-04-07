import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"  # Replace with your actual Hugging Face token

import sys, gc, numpy as np, pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import warnings

warnings.filterwarnings("ignore")

os.environ["AG_MAX_MEMORY_USAGE_RATIO"] = "100.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
except ImportError: sys.exit("tabpfn_extensions uninstalled.")

# ================= Configuration =================
DATA_PATH = r'$(pwd)/data/LNP-447.CSV' 
RESULTS_DIR = r'$(pwd)/result'
Model_DIR = r'$(pwd)/model' # save_dir of the pkl model 

# 1. Embeddings of training data
EMBEDDING_FILES = {
    "1. Bionemo-FT (2565 D)": "/home/zengjunjie/tabpfn/LNP-447-ft-final.npy",
    #"2. Bionemo (2565 D)": "/home/zengjunjie/tabpfn/LNP-447-bionemo.npy",
    #"3. Grover_base (16005 D)": "/home/zengjunjie/tabpfn/LNP-447-grover.npz",
    #"4. Grover_large (24005 D)": "/home/zengjunjie/tabpfn/LNP-447-grover.npz",
    #"5. Rdkit (10245 D)": "/home/zengjunjie/tabpfn/LNP-447-rdkit.csv",
    #"6. Padel_descriptor (4410 D)": "/home/zengjunjie/tabpfn/LNP-447-padel.csv"
}

# 2. Embeddings of new samples for PCA
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-neibu/LNP-internal-ft-final.npy'
NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-文献配方/LipidNemo_embedding/LNP-external-ft-final.npy'
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-external-grover.npz'
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-external-rdkit.csv'
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-external-padel.csv'
#NEW_DATA_NPY_PATH = ""

TARGET_COLUMN = "Organ"
TARGET_CLASSES = ["Liver", "Lung", "Spleen", "None"]

# PCA
PCA_N_COMPONENTS = 80  
RANDOM_SEED = 42


TabPFN_PARAMS = {
    'ignore_pretraining_limits': True, 
    'random_state': RANDOM_SEED, 
    'n_estimators': 4,        
    'max_time': 900,          
    'device': 'cuda:0'        
}
# =================================================
def data_load(file_path):
    """
    Smart loading function: automatically recognize .npy, .npz, .csv
    """
    if not os.path.exists(file_path):
        return None
        
    ext = os.path.splitext(file_path)[-1].lower()
    
    print(f"   Detected format {ext}: Reading...", end="")
    
    try:
        if ext == '.npy':
            data = np.load(file_path)
        
        elif ext == '.npz':
            # .npz is a compressed archive, usually we take the first array
            with np.load(file_path) as loaded:
                # Try to get the default 'arr_0', or the first key in the archive
                keys = list(loaded.keys())
                if 'arr_0' in keys:
                    data = loaded['arr_0']
                else:
                    data = loaded[keys[0]] # Take the first one
                print(f" (Extracted key: {keys[0]})", end="")
                
        elif ext == '.csv':
            # .csv usually has no header, pure numeric matrix
            # If your CSV has a header, please change header=None to header=0
            df = pd.read_csv(file_path, header=0)
            data = df.values.astype(np.float32)
            
        else:
            print(f" Unsupported format: {ext}")
            return None
            
        print(" Done.")
        return data
        
    except Exception as e:
        print(f" Failed to read: {e}")
        return None

def evaluate_model(name, filename, df_labels):
    print(f"\n" + "="*60)
    print(f" Processing: {name} (Transductive PCA Mode)")
    print("="*60)
    
    # --- 1. Load training data ---
    f_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(f_path): 
        print(f"   File not found: {filename}"); return None

    X_all = data_load(f_path)
    if X_all is None: return None

    # Align and clean data
    curr_df = df_labels.copy()
    if len(curr_df) != len(X_all):
        m = min(len(curr_df), len(X_all))
        curr_df, X_all = curr_df.iloc[:m], X_all[:m]
    
    curr_df[TARGET_COLUMN] = curr_df[TARGET_COLUMN].fillna('None').astype(str).str.strip().replace({'nan':'None', 'NaN':'None'})
    mask = curr_df[TARGET_COLUMN].isin(TARGET_CLASSES)
    y_filt = curr_df.loc[mask, TARGET_COLUMN].values
    X_filt = X_all[mask.values]

    # --- 2. Load new sample data ---
    if os.path.exists(NEW_DATA_NPY_PATH):
        X_new_raw = data_load(NEW_DATA_NPY_PATH)
        print(f"   Successfully loaded {X_new_raw.shape[0]} new samples for PCA alignment")
    else:
        print(f"   New sample file not found: {NEW_DATA_NPY_PATH}, will use standard PCA")
        X_new_raw = None

    # --- 3. Split training and testing sets ---
    # Here we only split once, no cross-validation, to save resources and focus on the final model
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_filt, y_filt, test_size=0.1, random_state=RANDOM_SEED, stratify=y_filt
    )
    print(f"   - Training set dimensions: {X_train_raw.shape}")

    # ================= 4. Transductive PCA =================
    print(f"\n Starting Transductive PCA processing (PCA={PCA_N_COMPONENTS})...")
    
    # Split Embedding (first N-5) and Ratios (last 5)
    emb_dim = X_train_raw.shape[1] - 5
    
    X_train_emb = X_train_raw[:, :emb_dim]
    X_test_emb  = X_test_raw[:, :emb_dim]
    if X_new_raw is not None:
        X_new_emb = X_new_raw[:, :emb_dim]
    
    # Aggressive cleaning (prevent NaN/Inf errors)
    X_train_emb = np.nan_to_num(X_train_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test_emb  = np.nan_to_num(X_test_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    if X_new_raw is not None:
        X_new_emb = np.nan_to_num(X_new_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)

    # A. Scaler (Standardization)
    # Note: Scaler should only fit the training set to maintain the baseline of data distribution
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb  = scaler.transform(X_test_emb)
    
    if X_new_raw is not None:
        X_new_emb = scaler.transform(X_new_emb)
        # 
        # Key point: Vertically stack the training set and new samples, feed them together to PCA fit
        X_for_pca_fit = np.vstack([X_train_emb, X_new_emb])
        print("   PCA Fit Strategy: Mixed mode (Training set + New samples)")
    else:
        X_for_pca_fit = X_train_emb
        print("   PCA Fit Strategy: Standard mode (Training set only)")

    # B. PCA Fit
    n_comp_actual = min(PCA_N_COMPONENTS, X_for_pca_fit.shape[0], X_for_pca_fit.shape[1])
    pca = PCA(n_components=n_comp_actual, random_state=RANDOM_SEED)
    pca.fit(X_for_pca_fit)  # Fit all data here

    # C. PCA Transform
    X_train_pca = pca.transform(X_train_emb)
    X_test_pca  = pca.transform(X_test_emb)
    
    # D. Concatenate Ratios back
    # Clean Ratio
    X_train_rat = np.nan_to_num(X_train_rat, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_rat  = np.nan_to_num(X_test_rat, nan=0.0, posinf=1.0, neginf=0.0)
    
    X_train_final = np.hstack([X_train_pca, X_train_rat])
    X_test_final  = np.hstack([X_test_pca, X_test_rat])
    
    print(f"   - Final training input dimensions: {X_train_final.shape}")

    # ================= 5. Train Model =================
    print(f" Starting TabPFN training...")
    
    # Note: The 'path' parameter is removed here because previous errors showed it is not supported
    # AutoGluon will automatically generate an 'AutogluonModels' folder in the current directory
    # We will not delete it for subsequent use
    final_clf = AutoTabPFNClassifier(**TabPFN_PARAMS)
    final_clf.fit(X_train_final, y_train)

    # 6. Predict test set
    y_pred = final_clf.predict(X_test_final)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   Internal test set accuracy: {test_acc:.4f}")

    # 7. Save model pipeline
    save_package = {
        "scaler": scaler,        
        "pca": pca,             
        "model": final_clf,      
        "n_components": PCA_N_COMPONENTS,
        "emb_dim": emb_dim,
        "target_classes": TARGET_CLASSES
    }
    
    save_name = f"lnp_transductive-文献配方7-Seed{RANDOM_SEED}_bionemo_{PCA_N_COMPONENTS}D.pkl"
    save_path = os.path.join(Model_DIR, save_name)
    joblib.dump(save_package, save_path)
    print(f" Model pipeline saved to: {save_path}")
    print(f" Important: The 'AutogluonModels' folder generated during training has been retained.")

    return {"Name": name, "Test_Acc": test_acc}

def main():
    if torch.cuda.is_available(): print(f" GPU ready: {torch.cuda.get_device_name(0)}")
    df_labels = pd.read_csv(DATA_PATH)

    results = []
    for name, fname in EMBEDDING_FILES.items():
        res = evaluate_model(name, fname, df_labels)
        if res: results.append(res)

    print("\n" + "="*80)
    print(f"{' FINAL RESULTS ':^80}")
    print("="*80)
    for r in results:
        print(f"{r['Name']:<25} | Test Acc: {r['Test_Acc']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
