import numpy as np
from m5_forecasting.metrics import smape, mase

def test_smape():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    result = smape(y_true, y_pred)
    assert 0 <= result <= 100

def test_mase():
    y_true = np.array([10, 12, 14, 16])
    y_pred = np.array([11, 13, 15, 17])
    naive = y_true[:-1]
    result = mase(y_true, y_pred, naive)
    assert result >= 0