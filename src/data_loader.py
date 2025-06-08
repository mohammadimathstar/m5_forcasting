import pandas as pd
from .config import RAW_DATA_DIR

def load_sales_data() -> pd.DataFrame:
    """
    Load the M5 sales training data.
    
    Returns:
        pd.DataFrame: sales_train_validation dataset
    """
    older_dates = [f'd_{i}' for i in range(1, 1850)]
    df = pd.read_csv(RAW_DATA_DIR / 'sales_train_validation.csv')
    df.drop(older_dates, axis=1, inplace=True)
    return df

def load_calendar_data() -> pd.DataFrame:
    """
    Load the calendar file with date mappings and events.

    Returns:
        pd.DataFrame: calendar.csv dataset
    """
    return pd.read_csv(RAW_DATA_DIR / 'calendar.csv')

def load_sell_prices() -> pd.DataFrame:
    """
    Load item price history data.

    Returns:
        pd.DataFrame: sell_prices.csv dataset
    """
    return pd.read_csv(RAW_DATA_DIR / 'sell_prices.csv')
