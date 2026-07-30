import pandas as pd
for c, name in [(0, 'Liver'), (1, 'Spleen'), (2, 'Lung')]:
    path = f"/home/zengjunjie/tabpfn/mechanism_analysis/PCA_embedding-analysis/seed22-combined_feature/target_class_{c}/mordred_shap_feature_importance_summary.csv"
    try:
        df = pd.read_csv(path).head(15)
        print(f"--- {name} (Class {c}) ---")
        print(", ".join(df['feature'].tolist()))
    except Exception as e:
        print(f"Error reading {c}: {e}")
