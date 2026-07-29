import pandas as pd
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from src.config import DATASET_PATH

def load_dataset():
    """
    Loads the dataset, separates features and target, and encodes the labels.
    """
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"[SUCCESS] Loaded Dataset: {df.shape[0]} records[cite: 1].")
    except FileNotFoundError:
        raise FileNotFoundError(f"[ERROR] Could not find '{DATASET_PATH}'.")

    # Isolate features and target
    X = df.drop(columns=['RiskLevel']).values
    y_raw = df['RiskLevel'].values
    feature_names = df.drop(columns=['RiskLevel']).columns

    # Encode labels to integers (0, 1, 2) required for XGBoost and Brier calculations
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    lb = LabelBinarizer()
    lb.fit(y)
    
    return X, y, feature_names, lb, df
