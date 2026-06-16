"""
data_collection.py

This script collects daily stock market data from Yahoo Finance for the selected companies
used in the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

The data supports analysis of unusual trading behaviour and anomalous market activity
around quarterly earnings announcements.
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE, RAW_STOCK_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_STOCK_DIR = PROJECT_ROOT / RAW_STOCK_DATA_PATH


REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Ticker",
]


def ensure_output_directory() -> None:
    """
    Create the stock price output directory if it does not already exist.
    """
    RAW_STOCK_DIR.mkdir(parents=True, exist_ok=True)


def get_yfinance_end_date(end_date: str) -> str:
    """
    yfinance treats the end date as exclusive.
    To include 2025-12-31, we add one extra day.
    """
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    adjusted_end_dt = end_dt + timedelta(days=1)
    return adjusted_end_dt.strftime("%Y-%m-%d")


def clean_stock_data(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean downloaded Yahoo Finance data and prepare it for saving.
    """
    if data.empty:
        return data

    data = data.reset_index()

    # Handle possible multi-index columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data["Date"] = pd.to_datetime(data["Date"]).dt.date
    data["Ticker"] = ticker

    # Keep only required columns
    available_columns = [col for col in REQUIRED_COLUMNS if col in data.columns]
    data = data[available_columns]

    # Sort and remove duplicate dates
    data = data.sort_values("Date")
    data = data.drop_duplicates(subset=["Date"], keep="first")

    # Ensure date range is strictly inside dissertation period
    data["Date"] = pd.to_datetime(data["Date"])
    data = data[
        (data["Date"] >= pd.to_datetime(START_DATE))
        & (data["Date"] <= pd.to_datetime(END_DATE))
    ]

    return data


def validate_stock_data(data: pd.DataFrame, ticker: str) -> bool:
    """
    Validate that the downloaded dataset contains all required columns and rows.
    """
    if data.empty:
        print(f"WARNING: No data downloaded for {ticker}.")
        return False

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]

    if missing_columns:
        print(f"WARNING: Missing columns for {ticker}: {missing_columns}")
        return False

    return True


def download_single_ticker(ticker: str) -> pd.DataFrame:
    """
    Download daily stock market data for one ticker.
    """
    print(f"Downloading {ticker}...")

    yf_end_date = get_yfinance_end_date(END_DATE)

    data = yf.download(
        ticker,
        start=START_DATE,
        end=yf_end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    data = clean_stock_data(data, ticker)

    if not validate_stock_data(data, ticker):
        return pd.DataFrame()

    output_file = RAW_STOCK_DIR / f"{ticker}_stock_data.csv"
    data.to_csv(output_file, index=False)

    print(f"Saved {ticker}_stock_data.csv with {len(data)} rows.")

    return data


def create_collection_summary(all_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary table for data collection.
    """
    summary_rows = []

    for data in all_dataframes:
        if data.empty:
            continue

        ticker = data["Ticker"].iloc[0]

        summary_rows.append(
            {
                "Ticker": ticker,
                "Rows": len(data),
                "Start_Date": data["Date"].min().strftime("%Y-%m-%d"),
                "End_Date": data["Date"].max().strftime("%Y-%m-%d"),
                "Missing_Values_Total": int(data.isna().sum().sum()),
                "Output_File": f"{ticker}_stock_data.csv",
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary_file = RAW_STOCK_DIR / "data_collection_summary.csv"
    summary.to_csv(summary_file, index=False)

    print("Data collection summary saved.")

    return summary


def save_combined_dataset(all_dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Save one combined stock dataset containing all tickers.
    """
    valid_dataframes = [df for df in all_dataframes if not df.empty]

    if not valid_dataframes:
        print("WARNING: No valid data available to combine.")
        return pd.DataFrame()

    combined_data = pd.concat(valid_dataframes, ignore_index=True)

    combined_data = combined_data.sort_values(["Ticker", "Date"])

    output_file = RAW_STOCK_DIR / "all_stock_data.csv"
    combined_data.to_csv(output_file, index=False)

    print(f"Combined dataset saved with {len(combined_data)} rows.")

    return combined_data


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting stock market data collection")
    print("Project focus: unusual trading behaviour around earnings announcements")
    print("=" * 70)

    ensure_output_directory()

    all_dataframes = []

    for ticker in TICKERS:
        ticker_data = download_single_ticker(ticker)
        all_dataframes.append(ticker_data)

    save_combined_dataset(all_dataframes)
    summary = create_collection_summary(all_dataframes)

    print("\nData Collection Summary:")
    print(summary)

    print("\nData collection completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()