
import pandas as pd
import numpy as np
from m5_forecasting.model import train_model

def test_train_model():
    X_train = pd.DataFrame(np.random.rand(100, 3), columns=['f1', 'f2', 'f3'])
    y_train = pd.Series(np.random.rand(100))
    X_valid = pd.DataFrame(np.random.rand(20, 3), columns=['f1', 'f2', 'f3'])
    y_valid = pd.Series(np.random.rand(20))

    model = train_model(X_train, y_train, X_valid, y_valid)
    assert model is not None
    assert hasattr(model, "predict")

