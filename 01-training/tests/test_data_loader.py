
import pandas as pd
import pytest
from m5_forecasting import data_loader

def test_load_sales_data(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({"id": [1], "d_1": [5]}))
    df = data_loader.load_sales_data()
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns

def test_load_calendar_data(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({"date": ["2020-01-01"], "d": ["d_1"]}))
    df = data_loader.load_calendar_data()
    assert "date" in df.columns

def test_load_sell_prices(monkeypatch):
    monkeypatch.setattr(pd, "read_csv", lambda path: pd.DataFrame({"store_id": ["CA_1"], "item_id": ["FOODS_1_001"], "sell_price": [2.5]}))
    df = data_loader.load_sell_prices()
    assert "sell_price" in df.columns

