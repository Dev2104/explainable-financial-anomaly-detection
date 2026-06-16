"""
earnings_loader.py

This script loads and validates quarterly earnings announcement dates for the selected
companies used in the dissertation project.

The earnings dates are used to construct 10-trading-day pre-earnings event windows
for evaluating unusual trading behaviour and anomalous market activity.
"""

from pathlib import Path

import pandas as pd

from config import TICKERS, RAW_EARNINGS_DATA_PATH, PROCESSED_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_EARNINGS_DIR = PROJECT_ROOT / RAW_EARNINGS_DATA_PATH
PROCESSED_DIR = PROJECT_ROOT / PROCESSED_DATA_PATH


REQUIRED_COLUMNS = [
    "Ticker",
    "Earnings_Date",
    "Source",
]


def ensure_output_directory() -> None:
    """
    Create the processed output directory if it does not already exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_single_earnings_file(ticker: str) -> pd.DataFrame:
    """
    Load one earnings date CSV file for a given ticker.
    """
    file_path = RAW_EARNINGS_DIR / f"{ticker}_earnings_dates.csv"

    if not file_path.exists():
        print(f"WARNING: Missing earnings date file for {ticker}: {file_path}")
        return pd.DataFrame()

    data = pd.read_csv(file_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]

    if missing_columns:
        print(f"WARNING: Missing columns in {ticker} earnings file: {missing_columns}")
        return pd.DataFrame()

    data["Ticker"] = data["Ticker"].astype(str).str.upper().str.strip()
    data["Earnings_Date"] = pd.to_datetime(data["Earnings_Date"], errors="coerce")
    data["Source"] = data["Source"].astype(str).str.strip()

    data = data.dropna(subset=["Ticker", "Earnings_Date"])
    data = data[data["Ticker"] == ticker]

    data = data.drop_duplicates(subset=["Ticker", "Earnings_Date"])
    data = data.sort_values(["Ticker", "Earnings_Date"])

    return data


def load_all_earnings_dates() -> pd.DataFrame:
    """
    Load all earnings date files and combine them into one dataset.
    """
    all_dataframes = []

    for ticker in TICKERS:
        print(f"Loading earnings dates for {ticker}...")
        ticker_data = load_single_earnings_file(ticker)

        if ticker_data.empty:
            print(f"WARNING: No valid earnings dates found for {ticker}.")
        else:
            print(f"Loaded {len(ticker_data)} earnings dates for {ticker}.")
            all_dataframes.append(ticker_data)

    if not all_dataframes:
        print("WARNING: No earnings date data loaded.")
        return pd.DataFrame()

    combined_data = pd.concat(all_dataframes, ignore_index=True)
    combined_data = combined_data.sort_values(["Ticker", "Earnings_Date"])

    return combined_data


def create_earnings_summary(earnings_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of earnings date coverage by ticker.
    """
    if earnings_data.empty:
        return pd.DataFrame()

    summary = (
        earnings_data.groupby("Ticker")
        .agg(
            Earnings_Dates_Count=("Earnings_Date", "count"),
            First_Earnings_Date=("Earnings_Date", "min"),
            Last_Earnings_Date=("Earnings_Date", "max"),
            Sources_Used=("Source", lambda x: ", ".join(sorted(set(x)))),
        )
        .reset_index()
    )

    summary["First_Earnings_Date"] = summary["First_Earnings_Date"].dt.strftime("%Y-%m-%d")
    summary["Last_Earnings_Date"] = summary["Last_Earnings_Date"].dt.strftime("%Y-%m-%d")

    return summary


def save_outputs(earnings_data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Save combined earnings dates and summary table.
    """
    if earnings_data.empty:
        print("WARNING: No earnings data available to save.")
        return

    combined_output_file = PROCESSED_DIR / "all_earnings_dates.csv"
    summary_output_file = PROCESSED_DIR / "earnings_dates_summary.csv"

    earnings_data_to_save = earnings_data.copy()
    earnings_data_to_save["Earnings_Date"] = earnings_data_to_save["Earnings_Date"].dt.strftime(
        "%Y-%m-%d"
    )

    earnings_data_to_save.to_csv(combined_output_file, index=False)
    summary.to_csv(summary_output_file, index=False)

    print(f"Combined earnings dates saved to: {combined_output_file}")
    print(f"Earnings dates summary saved to: {summary_output_file}")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting earnings dates loading and validation")
    print("Purpose: preparing event dates for pre-earnings anomaly analysis")
    print("=" * 70)

    ensure_output_directory()

    earnings_data = load_all_earnings_dates()
    summary = create_earnings_summary(earnings_data)

    if not earnings_data.empty:
        print("\nEarnings Dates Summary:")
        print(summary)

    save_outputs(earnings_data, summary)

    print("\nEarnings dates loading completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()