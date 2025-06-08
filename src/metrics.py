
import numpy as np

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0  # avoid NaN
    return np.mean(diff) * 100

def mase(y_true, y_pred, y_naive):
    numerator = np.mean(np.abs(y_true - y_pred))
    denominator = np.mean(np.abs(y_true[1:] - y_true[:-1]))  # seasonal naive = lag 1
    return numerator / denominator if denominator != 0 else np.inf

def smape_lgb_metric(preds, train_data):
    y_true = train_data.get_label()
    return 'sMAPE', smape(y_true, preds), False  # False = lower is better
