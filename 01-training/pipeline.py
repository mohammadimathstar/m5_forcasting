from prefect import flow, task
from .data_loader import load_sales_data, load_calendar_data, load_sell_prices
from .feature_engineering import melt_sales_data, add_lag_features, add_rolling_features, add_date_features
from .model import train_model
from .config import PROCESSED_DATA_DIR
from .param_tunning import run_hyperopt

import pandas as pd

@task(name="Prepare_data", log_prints=True)
def prepare_data() -> pd.DataFrame:
    """
    Complete data preparation: loading, merging, and feature engineering.

    Returns:
        pd.DataFrame: Processed data ready for modeling
    """
    print("Loading data...")
    sales = load_sales_data()
    calendar = load_calendar_data()
    prices = load_sell_prices()

    print("Merging data...")
    sales = melt_sales_data(sales)
    sales = sales.merge(calendar, on='d', how='left')
    sales = sales.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    print("Adding features...")
    sales = add_lag_features(sales)
    sales = add_rolling_features(sales)
    sales = add_date_features(sales)

    print("Saving processed data...")
    sales.to_csv(PROCESSED_DATA_DIR / 'processed_data.csv')
    return sales

@task(name="Split_data", log_prints=True)
def split_data(df: pd.DataFrame):
    """
    Split dataset into training and validation sets.

    Args:
        df (pd.DataFrame): Full preprocessed dataset

    Returns:
        Tuple: (X_train, y_train, X_valid, y_valid)
    """
    features = [
        'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
        'sell_price', 'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_mean_28',
        'weekday', 'week', 'month', 'year'
    ]
    for col in ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']:
        df[col] = df[col].astype('category')

    train_mask = df['d'].isin([f'd_{i}' for i in range(1850, 1900)])
    valid_mask = df['d'].isin([f'd_{i}' for i in range(1900, 1915)])

    X_train = df[train_mask][features]
    y_train = df[train_mask]['sales']
    X_valid = df[valid_mask][features]
    y_valid = df[valid_mask]['sales']

    return X_train, y_train, X_valid, y_valid

@flow(name="m5_pipeline", log_prints=True)
def m5_pipelines():
    
    """
    Execute full M5 forecasting pipeline.
    """
    print("Starting pipeline...")
    
    df = prepare_data()

    print("Splitting data...")
    X_train, y_train, X_valid, y_valid = split_data(df)
    
    print("Hyper-parameter search...")
    best_params = run_hyperopt(X_train, y_train, X_valid, y_valid, max_evals=2)

    print("Training model...")
    model = train_model(X_train, y_train, X_valid, y_valid, best_params)
    return model


if __name__ == "__main__":
    m5_pipelines()
