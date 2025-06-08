# m5_forecasting/tests/test_pipeline.py

import pandas as pd
from m5_forecasting.pipeline import split_data

def test_split_data():
    df = pd.DataFrame({
        'id': ['A'] * 50,
        'sales': list(range(50)),
        'd': ['d_' + str(i + 1500) for i in range(50)],
        'item_id': ['item_1'] * 50,
        'dept_id': ['dept_1'] * 50,
        'cat_id': ['cat_1'] * 50,
        'store_id': ['store_1'] * 50,
        'state_id': ['CA'] * 50,
        'sell_price': [1.0] * 50,
        'lag_7': [1.0] * 50,
        'lag_28': [1.0] * 50,
        'rolling_mean_7': [1.0] * 50,
        'rolling_mean_28': [1.0] * 50,
        'weekday': [1] * 50,
        'week': [1] * 50,
        'month': [1] * 50,
        'year': [2020] * 50,
    })

    X_train, y_train, X_valid, y_valid = split_data.fn(df)
    assert not X_train.empty
    assert not y_train.empty
    assert list(X_train.columns) == list(X_valid.columns)

