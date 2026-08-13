from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed. It will be skipped.")


warnings.filterwarnings('ignore')

def get_models(random_state=42):
    """
    Returns a dictionary of uninitialized baseline models.
    """
    models = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "Support Vector Machine": Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=random_state))
        ])
    }
    
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(random_state=random_state, eval_metric='mlogloss', use_label_encoder=False)
        
    return models
