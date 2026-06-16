"""
feature_engineering.py

This script creates financial trading features for the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

The features are used to detect unusual trading behaviour and anomalous market activity
around quarterly earnings announcements using unsupervised anomaly detection models.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_PATH, OUTPUT_TABLES_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / PROCESSED_DATA_PATH
OUTPUT_TABLES_DIR = PROJECT_ROOT / OUTPUT_TABLES_PATH

EVENT_WINDOW_DATA_FILE = PROCESSED_DIR / "event_window_dataset.csv"
EARNINGS_DATES_FILE = PROCESSED_DIR / "all_earnings_dates.csv"

FINAL_FEATURE_DATA_FILE = PROCESSED_DIR / "final_feature_dataset.csv"
FEATURE_SUMMARY_FILE = OUTPUT_TABLES_DIR / "feature_engineering_summary.csv"
MISSING_VALUES_FILE = OUTPUT_TABLES_DIR / "feature_missing_values_summary.csv"


FEATURE_COLUMNS = [
    "Daily_Return",
    "Relative_Volume",
    "Rolling_Volatility_10D",
    "Moving_Average_10D",
    "Volume_Change_Pct",
    "Price_Change_Pct",
    "Distance_From_Earnings_Event",
]


def ensure_output_directories() -> None:
    """
    Create required output directories if they do not already exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_event_window_data() -> pd.DataFrame:
    """
    Load event-window dataset created in Step 4.
    """
    if not EVENT_WINDOW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Event window dataset not found: {EVENT_WINDOW_DATA_FILE}. "
            "Please run src/event_windows.py first."
        )

    data = pd.read_csv(EVENT_WINDOW_DATA_FILE)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Ticker",
        "Event_Window",
        "Event_Earnings_Date",
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in event-window data: {missing_columns}")

    data["Date"] = pd.to_datetime(data["Date"])
    data["Ticker"] = data["Ticker"].astype(str).str.upper().str.strip()
    data["Event_Window"] = data["Event_Window"].astype(int)

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return data


def load_earnings_dates() -> pd.DataFrame:
    """
    Load earnings dates for calculating distance from earnings event.
    """
    if not EARNINGS_DATES_FILE.exists():
        raise FileNotFoundError(
            f"Earnings dates file not found: {EARNINGS_DATES_FILE}. "
            "Please run src/earnings_loader.py first."
        )

    earnings_data = pd.read_csv(EARNINGS_DATES_FILE)

    required_columns = ["Ticker", "Earnings_Date"]

    missing_columns = [col for col in required_columns if col not in earnings_data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in earnings data: {missing_columns}")

    earnings_data["Ticker"] = earnings_data["Ticker"].astype(str).str.upper().str.strip()
    earnings_data["Earnings_Date"] = pd.to_datetime(earnings_data["Earnings_Date"])

    earnings_data = earnings_data.sort_values(["Ticker", "Earnings_Date"]).reset_index(
        drop=True
    )

    return earnings_data


def calculate_distance_from_earnings_event(
    ticker_data: pd.DataFrame,
    ticker_earnings_dates: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the minimum trading-day distance from the nearest earnings announcement.

    The distance is based on trading-day index position, not calendar-day distance.
    This is consistent with the 10-trading-day pre-earnings event window design.
    """
    ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

    if ticker_earnings_dates.empty:
        return pd.Series([np.nan] * len(ticker_data))

    trading_dates = ticker_data["Date"].reset_index(drop=True)

    earnings_positions = []

    for earnings_date in ticker_earnings_dates["Earnings_Date"]:
        matching_positions = trading_dates[trading_dates >= earnings_date].index

        if len(matching_positions) > 0:
            earnings_positions.append(matching_positions[0])

    if not earnings_positions:
        return pd.Series([np.nan] * len(ticker_data))

    distances = []

    for row_position in range(len(ticker_data)):
        nearest_distance = min(abs(row_position - event_position) for event_position in earnings_positions)
        distances.append(nearest_distance)

    return pd.Series(distances)


def engineer_features_for_ticker(
    ticker_data: pd.DataFrame,
    ticker_earnings_dates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create all required financial features for one ticker.
    """
    ticker_data = ticker_data.copy()
    ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

    # Daily return based on adjusted closing price
    ticker_data["Daily_Return"] = ticker_data["Adj Close"].pct_change()

    # Relative volume compared to previous 10 trading days
    ticker_data["Relative_Volume"] = (
        ticker_data["Volume"] / ticker_data["Volume"].shift(1).rolling(window=10).mean()
    )

    # 10-day rolling volatility of daily returns
    ticker_data["Rolling_Volatility_10D"] = (
        ticker_data["Daily_Return"].rolling(window=10).std()
    )

    # 10-day moving average of adjusted closing price
    ticker_data["Moving_Average_10D"] = ticker_data["Adj Close"].rolling(window=10).mean()

    # Daily percentage change in trading volume
    ticker_data["Volume_Change_Pct"] = ticker_data["Volume"].pct_change() * 100

    # Daily percentage change in adjusted closing price
    ticker_data["Price_Change_Pct"] = ticker_data["Adj Close"].pct_change() * 100

    # Trading-day distance from nearest earnings event
    ticker_data["Distance_From_Earnings_Event"] = calculate_distance_from_earnings_event(
        ticker_data=ticker_data,
        ticker_earnings_dates=ticker_earnings_dates,
    )

    return ticker_data


def create_all_features(
    event_window_data: pd.DataFrame,
    earnings_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create features for all tickers and combine into one final dataset.
    """
    all_feature_dataframes = []

    tickers = sorted(event_window_data["Ticker"].unique())

    for ticker in tickers:
        print(f"Creating features for {ticker}...")

        ticker_data = event_window_data[event_window_data["Ticker"] == ticker].copy()
        ticker_earnings_dates = earnings_data[earnings_data["Ticker"] == ticker].copy()

        ticker_feature_data = engineer_features_for_ticker(
            ticker_data=ticker_data,
            ticker_earnings_dates=ticker_earnings_dates,
        )

        all_feature_dataframes.append(ticker_feature_data)

        print(f"{ticker}: features created for {len(ticker_feature_data)} rows.")

    final_data = pd.concat(all_feature_dataframes, ignore_index=True)
    final_data = final_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return final_data


def clean_feature_dataset(feature_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean feature dataset by replacing infinite values and removing rows with missing
    values in required feature columns.
    """
    missing_before = feature_data[FEATURE_COLUMNS].isna().sum().reset_index()
    missing_before.columns = ["Feature", "Missing_Values_Before_Cleaning"]

    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)

    rows_before = len(feature_data)

    cleaned_data = feature_data.dropna(subset=FEATURE_COLUMNS).copy()

    rows_after = len(cleaned_data)
    rows_removed = rows_before - rows_after

    missing_after = cleaned_data[FEATURE_COLUMNS].isna().sum().reset_index()
    missing_after.columns = ["Feature", "Missing_Values_After_Cleaning"]

    missing_summary = missing_before.merge(missing_after, on="Feature", how="left")

    print(f"\nRows before feature cleaning: {rows_before}")
    print(f"Rows after feature cleaning: {rows_after}")
    print(f"Rows removed due to rolling/pct-change missing values: {rows_removed}")

    return cleaned_data, missing_summary


def create_feature_summary(cleaned_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create descriptive summary statistics for engineered features.
    """
    summary = cleaned_data[FEATURE_COLUMNS].describe().T.reset_index()
    summary = summary.rename(columns={"index": "Feature"})

    summary = summary[
        [
            "Feature",
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]

    return summary


def validate_final_dataset(cleaned_data: pd.DataFrame) -> None:
    """
    Validate final feature dataset before saving.
    """
    required_columns = [
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Event_Window",
        "Event_Earnings_Date",
    ] + FEATURE_COLUMNS

    missing_columns = [col for col in required_columns if col not in cleaned_data.columns]

    if missing_columns:
        raise ValueError(f"Final dataset is missing columns: {missing_columns}")

    if cleaned_data.empty:
        raise ValueError("Final feature dataset is empty.")

    remaining_missing_values = cleaned_data[FEATURE_COLUMNS].isna().sum().sum()

    if remaining_missing_values > 0:
        raise ValueError(
            f"Final feature dataset still has {remaining_missing_values} missing feature values."
        )

    print("\nFinal dataset validation passed.")
    print(f"Final rows: {len(cleaned_data)}")
    print(f"Tickers: {sorted(cleaned_data['Ticker'].unique())}")
    print(f"Feature columns: {FEATURE_COLUMNS}")


def save_outputs(
    cleaned_data: pd.DataFrame,
    feature_summary: pd.DataFrame,
    missing_summary: pd.DataFrame,
) -> None:
    """
    Save final feature dataset and summary tables.
    """
    data_to_save = cleaned_data.copy()
    data_to_save["Date"] = data_to_save["Date"].dt.strftime("%Y-%m-%d")

    data_to_save.to_csv(FINAL_FEATURE_DATA_FILE, index=False)
    feature_summary.to_csv(FEATURE_SUMMARY_FILE, index=False)
    missing_summary.to_csv(MISSING_VALUES_FILE, index=False)

    print(f"\nFinal feature dataset saved to: {FINAL_FEATURE_DATA_FILE}")
    print(f"Feature summary saved to: {FEATURE_SUMMARY_FILE}")
    print(f"Missing values summary saved to: {MISSING_VALUES_FILE}")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting feature engineering")
    print("Purpose: preparing ML-ready data for unsupervised anomaly detection")
    print("=" * 70)

    ensure_output_directories()

    event_window_data = load_event_window_data()
    earnings_data = load_earnings_dates()

    print(f"Loaded event-window dataset rows: {len(event_window_data)}")
    print(f"Loaded earnings date rows: {len(earnings_data)}")

    feature_data = create_all_features(
        event_window_data=event_window_data,
        earnings_data=earnings_data,
    )

    cleaned_data, missing_summary = clean_feature_dataset(feature_data)

    feature_summary = create_feature_summary(cleaned_data)

    print("\nFeature Engineering Summary:")
    print(feature_summary)

    print("\nMissing Values Summary:")
    print(missing_summary)

    validate_final_dataset(cleaned_data)

    save_outputs(
        cleaned_data=cleaned_data,
        feature_summary=feature_summary,
        missing_summary=missing_summary,
    )

    print("\nFeature engineering completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()