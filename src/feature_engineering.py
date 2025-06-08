import pandas as pd

def melt_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format sales data to long-format.

    Args:
        sales_df (pd.DataFrame): Raw sales data

    Returns:
        pd.DataFrame: Long-format sales data
    """
    return sales_df.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d',
        value_name='sales'
    )

def add_lag_features(df: pd.DataFrame, lags: list=[7, 28]) -> pd.DataFrame:
    """
    Add lag-based features for time series modeling.

    Args:
        df (pd.DataFrame): Input data
        lags (list): Lag periods to compute

    Returns:
        pd.DataFrame: Data with lag features
    """
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, windows: list=[7, 28]) -> pd.DataFrame:
    """
    Add rolling statistics to sales data.

    Args:
        df (pd.DataFrame): Data with lag features
        windows (list): Rolling window sizes

    Returns:
        pd.DataFrame: Data with rolling mean features
    """
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('id')['lag_28'].transform(lambda x: x.rolling(window).mean())
    return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard date features for modeling seasonality.

    Args:
        df (pd.DataFrame): Sales data with 'date' column

    Returns:
        pd.DataFrame: Data with date features
    """
    df['date'] = pd.to_datetime(df['date'])
    df['weekday'] = df['date'].dt.weekday
    df['week'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    return df

