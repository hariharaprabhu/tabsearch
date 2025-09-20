"""
datasets.py

Helper functions to quickly load demo datasets for mixed-data-vectorizer.
"""

import pandas as pd

_SP500_URL = "https://raw.githubusercontent.com/hariharaprabhu/mixed-data-vectorizer/main/Examples/Similar%20Stock%20Tickers/sp500_companies.csv"


def load_sp500_demo() -> pd.DataFrame:
    """
    Load the S&P 500 demo dataset (7 mixed columns, ~500 rows).

    This dataset includes:
    - Symbol (ID/index)
    - Sector (categorical)
    - Industry (categorical)
    - Currentprice (numerical)
    - Marketcap (numerical)
    - Fulltimeemployees (numerical)
    - Longbusinesssummary (text)

    Returns
    -------
    pd.DataFrame
        A DataFrame with ~500 rows and 7 columns.

    Example
    -------
    >>> from TabFusion.datasets import load_sp500_demo
    >>> df = load_sp500_demo()
    >>> df.head()
    """
    return pd.read_csv(_SP500_URL)
