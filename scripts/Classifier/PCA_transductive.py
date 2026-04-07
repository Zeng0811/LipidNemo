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
    "1. Bionemo-FT": "/home/zengjunjie/tabpfn/LNP-447-ft-final.npy",
    #"2. Bionemo": "/home/zengjunjie/tabpfn/LNP-448-bionemo.npy",
    #"3. Grover": "/home/zengjunjie/tabpfn/LNP-448-grover.npz",
    #"5. Rdkit": "/home/zengjunjie/tabpfn/LNP-448-rdkit.csv",
    #"6. Padel_descriptor": "/home/zengjunjie/tabpfn/LNP-448-padel.csv"
}

# 2. Embeddings of new samples for PCA
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-neibu/LNP-内部-ft-final.npy'
NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-文献配方/LipidNemo_embedding/LNP-文献配方-7.npy'
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-内部-grover_large.npz'
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-内部-rdkit_fp.csv'
#NEW_DATA_NPY_PATH = r'/home/zengjunjie/tabpfn/LNP-内部-padel_fp.csv'
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
    智能加载函数：自动识别 .npy, .npz, .csv
    """
    if not os.path.exists(file_path):
        return None
        
    ext = os.path.splitext(file_path)[-1].lower()
    
    print(f"   📂 检测到格式 {ext}: 正在读取...", end="")
    
    try:
        if ext == '.npy':
            data = np.load(file_path)
        
        elif ext == '.npz':
            # .npz 是压缩包，通常我们取第一个数组
            with np.load(file_path) as loaded:
                # 尝试获取默认的 'arr_0'，或者是压缩包里的第一个键
                keys = list(loaded.keys())
                if 'arr_0' in keys:
                    data = loaded['arr_0']
                else:
                    data = loaded[keys[0]] # 取第一个
                print(f" (提取键: {keys[0]})", end="")
                
        elif ext == '.csv':
            # .csv 通常没有表头，纯数字矩阵
            # 如果你的CSV有表头(header)，请把 header=None 改为 header=0
            df = pd.read_csv(file_path, header=0)
            data = df.values.astype(np.float32)
            
        else:
            print(f" ❌ 不支持的格式: {ext}")
            return None
            
        print(" ✅")
        return data
        
    except Exception as e:
        print(f" ❌ 读取失败: {e}")
        return None

def evaluate_model(name, filename, df_labels):
    print(f"\n" + "="*60)
    print(f"🔄 正在处理: {name} (直推式 PCA 模式)")
    print("="*60)
    
    # --- 1. 加载训练数据 ---
    f_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(f_path): 
        print(f"   ⚠️ 文件不存在: {filename}"); return None

    X_all = data_load(f_path)
    if X_all is None: return None

    # 对齐与清洗数据
    curr_df = df_labels.copy()
    if len(curr_df) != len(X_all):
        m = min(len(curr_df), len(X_all))
        curr_df, X_all = curr_df.iloc[:m], X_all[:m]
    
    curr_df[TARGET_COLUMN] = curr_df[TARGET_COLUMN].fillna('None').astype(str).str.strip().replace({'nan':'None', 'NaN':'None'})
    mask = curr_df[TARGET_COLUMN].isin(TARGET_CLASSES)
    y_filt = curr_df.loc[mask, TARGET_COLUMN].values
    X_filt = X_all[mask.values]

    # --- 2. 加载新样本数据 ---
    if os.path.exists(NEW_DATA_NPY_PATH):
        X_new_raw = data_load(NEW_DATA_NPY_PATH)
        print(f"   ✅ 成功加载 {X_new_raw.shape[0]} 个新样本用于 PCA 辅助定位")
    else:
        print(f"   ⚠️ 未找到新样本文件: {NEW_DATA_NPY_PATH}，将使用普通 PCA")
        X_new_raw = None

    # --- 3. 划分训练集和测试集 ---
    # 这里我们只划分一次，不做交叉验证，以节省资源并专注最终模型
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_filt, y_filt, test_size=0.1, random_state=RANDOM_SEED, stratify=y_filt
    )
    print(f"   - 训练集维度: {X_train_raw.shape}")

    # ================= 4. 直推式 PCA (Transductive PCA) =================
    print(f"\n🚀 开始直推式 PCA 处理 (PCA={PCA_N_COMPONENTS})...")
    
    # 拆分 Embedding (前N-5) 和 Ratios (后5)
    emb_dim = X_train_raw.shape[1] - 5
    
    X_train_emb = X_train_raw[:, :emb_dim]
    X_test_emb  = X_test_raw[:, :emb_dim]
    if X_new_raw is not None:
        X_new_emb = X_new_raw[:, :emb_dim]
    
    X_train_rat = X_train_raw[:, emb_dim:]
    X_test_rat  = X_test_raw[:, emb_dim:]

    # 强力清洗 (防止 NaN/Inf 报错)
    X_train_emb = np.nan_to_num(X_train_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    X_test_emb  = np.nan_to_num(X_test_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)
    if X_new_raw is not None:
        X_new_emb = np.nan_to_num(X_new_emb, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)

    # A. Scaler (标准化)
    # 注意：Scaler 应该只 Fit 训练集，保持数据分布的基准
    scaler = StandardScaler()
    X_train_emb = scaler.fit_transform(X_train_emb)
    X_test_emb  = scaler.transform(X_test_emb)
    
    if X_new_raw is not None:
        X_new_emb = scaler.transform(X_new_emb)
        # 
        # 🔥 关键点：将训练集和新样本垂直拼接，一起喂给 PCA fit 🔥
        X_for_pca_fit = np.vstack([X_train_emb, X_new_emb])
        print("   👉 PCA Fit 策略: 混合模式 (训练集 + 新样本)")
    else:
        X_for_pca_fit = X_train_emb
        print("   👉 PCA Fit 策略: 普通模式 (仅训练集)")

    # B. PCA Fit
    n_comp_actual = min(PCA_N_COMPONENTS, X_for_pca_fit.shape[0], X_for_pca_fit.shape[1])
    pca = PCA(n_components=n_comp_actual, random_state=RANDOM_SEED)
    pca.fit(X_for_pca_fit)  # 这里 Fit 了所有数据

    # C. PCA Transform
    X_train_pca = pca.transform(X_train_emb)
    X_test_pca  = pca.transform(X_test_emb)
    
    # D. 拼接回 Ratios
    # 清洗 Ratio
    X_train_rat = np.nan_to_num(X_train_rat, nan=0.0, posinf=1.0, neginf=0.0)
    X_test_rat  = np.nan_to_num(X_test_rat, nan=0.0, posinf=1.0, neginf=0.0)
    
    X_train_final = np.hstack([X_train_pca, X_train_rat])
    X_test_final  = np.hstack([X_test_pca, X_test_rat])
    
    print(f"   - 最终训练输入维度: {X_train_final.shape}")

    # ================= 5. 训练模型 =================
    print(f"🚀 开始训练 TabPFN...")
    
    # ⚠️ 注意：这里去掉了 'path' 参数，因为之前的报错显示它不支持
    # AutoGluon 会自动在当前目录下生成 'AutogluonModels' 文件夹
    # 我们不会删除它，以便后续使用
    final_clf = AutoTabPFNClassifier(**TabPFN_PARAMS)
    final_clf.fit(X_train_final, y_train)

    # 6. 预测测试集
    y_pred = final_clf.predict(X_test_final)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   🏆 内部测试集准确率: {test_acc:.4f}")

    # 7. 💾 保存模型流水线
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
    print(f"✅ 模型流水线已保存至: {save_path}")
    print(f"⚠️  重要：训练生成的 'AutogluonModels' 文件夹已保留。")

    return {"Name": name, "Test_Acc": test_acc}

def main():
    if torch.cuda.is_available(): print(f"✅ GPU 就绪: {torch.cuda.get_device_name(0)}")
    df_labels = pd.read_csv(DATA_PATH)

    results = []
    for name, fname in EMBEDDING_FILES.items():
        res = evaluate_model(name, fname, df_labels)
        if res: results.append(res)

    print("\n" + "="*80)
    print(f"{'🏆 FINAL RESULTS 🏆':^80}")
    print("="*80)
    for r in results:
        print(f"{r['Name']:<25} | Test Acc: {r['Test_Acc']:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()