import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import warnings
from copy import deepcopy

from src.config import TIERS, N_SPLITS, RANDOM_STATE, DEGRADATION_SEEDS, PROJECT_ROOT
from src.preprocessing import load_dataset
from src.models import get_models
from src.degradation import DataDegradationEngine
from src.metrics import calculate_metrics

warnings.filterwarnings('ignore')

def run_ablation_study():
    # Load dataset
    X, y, feature_names, lb, le, df_raw = load_dataset()
    engine = DataDegradationEngine(random_state=RANDOM_STATE)
    
    ablation_tiers = [
        "Pristine Baseline",
        "Ablation: Missingness Only",
        "Ablation: BP Drift Only",
        "Ablation: Temp Bias Only"
    ]
    
    results = []
    models_dict = get_models(random_state=RANDOM_STATE)
    
    print("\nStarting Ablation Study...")
    
    for tier_name in ablation_tiers:
        tier_config = TIERS[tier_name]
        print(f"Evaluating: {tier_name}")
        
        # Using 3 seeds to balance robustness and execution time
        for seed in DEGRADATION_SEEDS[:3]: 
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(df_raw, y)):
                df_train = df_raw.iloc[train_idx].copy()
                df_test  = df_raw.iloc[test_idx].copy()
                
                y_train = y[train_idx]
                y_test  = y[test_idx]
                y_test_bin = lb.transform(y_test)
                
                # Apply degradation to TEST set only (simulate real-world failure at inference)
                engine_seed = DataDegradationEngine(random_state=seed)
                df_test_corrupted = engine_seed.apply_degradation(df_test, tier_config)
                
                # Extract X
                X_train = df_train.drop(columns=['RiskLevel']).values
                X_test_corr = df_test_corrupted.drop(columns=['RiskLevel']).values
                
                # Impute missing values for models that don't support NaNs natively
                imputer = SimpleImputer(strategy='median')
                imputer.fit(X_train)
                
                for model_name, base_model in models_dict.items():
                    model = deepcopy(base_model)
                    
                    # XGBoost handles missing natively; others need imputation
                    if model_name != "XGBoost":
                        X_train_model = imputer.transform(X_train)
                        X_test_model = imputer.transform(X_test_corr)
                    else:
                        X_train_model = X_train
                        X_test_model = X_test_corr
                        
                    model.fit(X_train_model, y_train)
                    y_pred = model.predict(X_test_model)
                    y_prob = model.predict_proba(X_test_model)
                    
                    acc, macro_f1, hr_recall, brier = calculate_metrics(y_test, y_pred, y_prob, y_test_bin)
                    
                    results.append({
                        "Tier": tier_name,
                        "Model": model_name,
                        "Seed": seed,
                        "Fold": fold,
                        "High Risk Recall": hr_recall,
                        "Accuracy": acc
                    })

    df_results = pd.DataFrame(results)
    
    # Aggregate results
    agg_results = df_results.groupby(["Tier", "Model"])["High Risk Recall"].mean().unstack()
    print("\n--- Average High Risk Recall ---")
    print(agg_results)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_results, x="Tier", y="High Risk Recall", hue="Model")
    plt.xticks(rotation=15)
    plt.title("Ablation Study: Impact of Isolated Degradations on High Risk Recall")
    plt.tight_layout()
    
    out_dir = os.path.join(PROJECT_ROOT, "results", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_study.png")
    plt.savefig(out_path)
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    run_ablation_study()
