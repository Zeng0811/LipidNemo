import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import pandas as pd
import joblib
import torch
import warnings
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

warnings.filterwarnings("ignore")

# ================= Configuration =================
# Model file path (the previously saved .pkl)
PIPELINE_PATH = r'$pwd/model/LNP-external-LipidNemo.pkl'
#PIPELINE_PATH = r'$pwd/model/LNP-internal-LipidNemo.pkl'   # Three internal formulations validated in vivo

# Embedding file for new data
# Note: This must be the raw features generated using the corresponding embedding method, prior to PCA
NEW_DATA_PATH = r'$pwd/data/LNP-external-ft-final.npy' 
#NEW_DATA_PATH = r'$pwd/data/LNP-internal-ft-final.npy' 

NEW_DATA_CSV = r'$pwd/data/LNP-external.CSV'
#NEW_DATA_CSV = r'$pwd/data/LNP-internal.CSV'

OUTPUT_PATH = r'$pwd/result'  # Prediction results output directory
# =======================================

def load_feature_file(file_path):
    """
    Intelligently load feature files, supporting .npy, .npz, and .csv formats.
    Returns: numpy.ndarray (N_samples, N_features)
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    # Extract file extension (converted to lowercase)
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"Loading: {os.path.basename(file_path)} ({ext}) ...", end="")

    try:
        # 1. Process .npy files (simplest format)
        if ext == '.npy':
            data = np.load(file_path)

        # 2. Process .npz files (dictionary-like format)
        elif ext == '.npz':
            with np.load(file_path) as loaded:
                # Retrieve all keys
                keys = list(loaded.keys())
                
                # Automatically search for probable feature keys
                if 'features' in keys:
                    target_key = 'features'  # Commonly used in models like Grover
                elif 'arr_0' in keys:
                    target_key = 'arr_0'     # Commonly used for default saves
                elif 'embeddings' in keys:
                    target_key = 'embeddings'
                else:
                    target_key = keys[0]     # If not found, default to the first key
                
                data = loaded[target_key]
                print(f" [Extracted key: '{target_key}']", end="")

        # 3. Process .csv files (text tabular format)
        elif ext == '.csv':
            df = pd.read_csv(file_path, header=0)
            data = df.values            
            # Ensure data type is float32
            data = data.astype(np.float32)

        else:
            print(f" Unsupported file format: {ext}")
            return None

        print(f" Success! Dimensions: {data.shape}")
        return data

    except Exception as e:
        print(f" Loading failed: {e}")
        return None

def predict_new_formulations():
    print(f"Loading model pipeline: {PIPELINE_PATH}")
    if not os.path.exists(PIPELINE_PATH):
        print("Model file not found. Please run the training script first."); return

    # 1. Load pipeline
    pipeline = joblib.load(PIPELINE_PATH)
    scaler = pipeline['scaler']
    pca = pipeline['pca']
    model = pipeline['model']
    emb_dim = pipeline['emb_dim']
    print("Model loaded successfully.")

    # 2. Load new data
    print(f"Loading new data: {NEW_DATA_PATH}")
    X_new = load_feature_file(NEW_DATA_PATH)
    print(f"   - Input dimensions: {X_new.shape}")

    if X_new.shape[1] != (emb_dim + 5):
        print(f"Dimension mismatch. The model expects {emb_dim+5} dimensions, but the input has {X_new.shape[1]} dimensions.")
        return

    # 3. Data preprocessing (must be identical to the training phase)
    print("Applying feature transformations...")
    
    # Split
    X_new_emb = X_new[:, :emb_dim]
    X_new_ratio = X_new[:, emb_dim:]

    # Clean
    X_new_emb = np.nan_to_num(X_new_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_new_ratio = np.nan_to_num(X_new_ratio, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    # Standardization (Transform ONLY, do not Fit)
    X_new_emb_scaled = scaler.transform(X_new_emb)

    # PCA dimensionality reduction (Transform ONLY, do not Fit)
    X_new_emb_pca = pca.transform(X_new_emb_scaled)

    # Concatenate
    X_final = np.hstack([X_new_emb_pca, X_new_ratio])
    X_final = np.nan_to_num(X_final, nan=0.0).astype(np.float32)
    
    print(f"   - Post-transformation dimensions: {X_final.shape} (Ready for prediction)")

    # 4. Predict
    print("Initiating prediction...")
    predictions = model.predict(X_final)
    probabilities = model.predict_proba(X_final)

    # 5. Display results
    print("\n" + "="*50)
    print(f"{'PREDICTION RESULTS':^50}")
    print("="*50)
    
    # Attempt to read names from the CSV file
    try:
        df_info = pd.read_csv(NEW_DATA_CSV)
        # Modification: Read the 'Ionizable lipid' column directly as sample names
        # Ensure the column name exactly matches the CSV header (case and space sensitive)
        if 'Ionizable lipid' in df_info.columns:
            names = df_info['Ionizable lipid'].astype(str).values
        else:
            # If the column name is not found, default to using the second column (index 1)
            names = df_info.iloc[:, 1].astype(str).values 
    except Exception as e:
        print(f"Failed to read names: {e}")
        names = [f"Sample_{i+1}" for i in range(len(predictions))]

    # Print the first 20 results
    print(f"{'Sample Name':<20} | {'Prediction':<10} | {'Confidence':<10}")
    print("-" * 50)
    
    classes = model.classes_
    for i in range(len(predictions)):
        pred_class = predictions[i]
        # Retrieve the probability of the predicted class
        conf = probabilities[i][list(classes).index(pred_class)]
        
        print(f"{names[i]:<20} | {pred_class:<10} | {conf:.2%}")
        
    # Save results to CSV
    result_df = pd.DataFrame({
        "Sample": names[:len(predictions)],
        "Predicted_Organ": predictions,
        "Confidence": [probabilities[i][list(classes).index(predictions[i])] for i in range(len(predictions))]
    })
    # Retain probabilities for all classes
    for idx, cls_name in enumerate(classes):
        result_df[f"Prob_{cls_name}"] = probabilities[:, idx]

    out_file = os.path.join(OUTPUT_PATH, "LNP-external-LipidNemo.csv")
    result_df.to_csv(out_file, index=False)
    print("="*50)
    print(f"Complete prediction results saved to: {out_file}")

if __name__ == "__main__":
    predict_new_formulations()
