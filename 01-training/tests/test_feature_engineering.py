import pandas as pd
import pytest
from m5_forecasting.feature_engineering import add_lag_features, add_rolling_features

def test_add_lag_features():
    df = pd.DataFrame({
        'id': ['A'] * 10,
        'sales': list(range(10))
    })
    result = add_lag_features(df.copy(), lags=[1])
    assert 'lag_1' in result.columns
    assert pd.isnull(result.loc[0, 'lag_1'])
    assert result.loc[1, 'lag_1'] == 0

def test_add_rolling_features():
    df = pd.DataFrame({
        'id': ['A'] * 10,
        'lag_28': list(range(10))
    })
    result = add_rolling_features(df.copy(), windows=[2])
    assert 'rolling_mean_2' in result.columns
    assert pd.isnull(result.loc[0, 'rolling_mean_2'])

