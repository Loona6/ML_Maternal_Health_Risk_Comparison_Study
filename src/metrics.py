import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

def calculate_metrics(y_true, y_pred, y_prob, y_true_bin):
    """
    Calculates accuracy, macro F1, high risk recall, and multi-class Brier Score.
    """
    acc = accuracy_score(y_true, y_pred)
    brier = np.mean(np.sum((y_true_bin - y_prob) ** 2, axis=1))
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    # 'high risk' is encoded as 0
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    high_risk_recall = recalls[0]
    
    return acc, macro_f1, high_risk_recall, brier
