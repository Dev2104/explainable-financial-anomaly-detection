"""
event_windows.py

This script creates 10-trading-day pre-earnings event windows for the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

The event window is used for unsupervised event-based evaluation of unusual trading
behaviour and anomalous market activity around quarterly earnings announcements.
"""

from pathlib import Path

import pandas as pd

from config import (
    TICKERS,
    EVENT_WINDOW_DAYS,
    RAW_STOCK_DATA_PATH,
    PROCESSED_DATA_PATH,
    OUTPUT_TABLES_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_STOCK_DIR = PROJECT_ROOT / RAW_STOCK_DATA_PATH
PROCESSED_DIR = PROJECT_ROOT / PROCESSED_DATA_PATH
OUTPUT_TABLES_DIR = PROJECT_ROOT / OUTPUT_TABLES_PATH

STOCK_DATA_FILE = RAW_STOCK_DIR / "all_stock_data.csv"
EARNINGS_DATA_FILE = PROCESSED_DIR / "all_earnings_dates.csv"


def ensure_output_directories() -> None:
    """
    Create required output directories if they do not already exist.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_stock_data() -> pd.DataFrame:
    """
    Load combined stock market data.
    """
    if not STOCK_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Stock data file not found: {STOCK_DATA_FILE}. "
            "Please run src/data_collection.py first."
        )

    stock_data = pd.read_csv(STOCK_DATA_FILE)

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Ticker",
    ]

    missing_columns = [col for col in required_columns if col not in stock_data.columns]

    if missing_columns:
        raise ValueError(f"Missing required stock data columns: {missing_columns}")

    stock_data["Date"] = pd.to_datetime(stock_data["Date"])
    stock_data["Ticker"] = stock_data["Ticker"].astype(str).str.upper().str.strip()

    stock_data = stock_data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return stock_data


def load_earnings_dates() -> pd.DataFrame:
    """
    Load combined earnings announcement dates.
    """
    if not EARNINGS_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Earnings data file not found: {EARNINGS_DATA_FILE}. "
            "Please run src/earnings_loader.py first."
        )

    earnings_data = pd.read_csv(EARNINGS_DATA_FILE)

    required_columns = ["Ticker", "Earnings_Date", "Source"]

    missing_columns = [col for col in required_columns if col not in earnings_data.columns]

    if missing_columns:
        raise ValueError(f"Missing required earnings date columns: {missing_columns}")

    earnings_data["Ticker"] = earnings_data["Ticker"].astype(str).str.upper().str.strip()
    earnings_data["Earnings_Date"] = pd.to_datetime(earnings_data["Earnings_Date"])

    earnings_data = earnings_data.sort_values(["Ticker", "Earnings_Date"]).reset_index(
        drop=True
    )

    return earnings_data


def create_event_windows_for_ticker(
    ticker_stock_data: pd.DataFrame,
    ticker_earnings_data: pd.DataFrame,
    ticker: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create 10-trading-day pre-earnings event windows for one ticker.

    Event_Window = 1 for the 10 trading days before each earnings announcement.
    Event_Window = 0 for all other trading days.
    """
    ticker_stock_data = ticker_stock_data.copy()
    ticker_stock_data = ticker_stock_data.sort_values("Date").reset_index(drop=True)

    ticker_stock_data["Event_Window"] = 0
    ticker_stock_data["Event_Earnings_Date"] = ""

    event_detail_rows = []

    for _, earnings_row in ticker_earnings_data.iterrows():
        earnings_date = earnings_row["Earnings_Date"]

        # Select only trading days strictly before the earnings announcement date
        previous_trading_days = ticker_stock_data[ticker_stock_data["Date"] < earnings_date]

        if previous_trading_days.empty:
            print(
                f"WARNING: No trading days found before earnings date "
                f"{earnings_date.date()} for {ticker}."
            )
            continue

        # Select the last 10 available trading days before the earnings date
        event_window_rows = previous_trading_days.tail(EVENT_WINDOW_DAYS)
        event_indices = event_window_rows.index

        ticker_stock_data.loc[event_indices, "Event_Window"] = 1
        ticker_stock_data.loc[event_indices, "Event_Earnings_Date"] = earnings_date.strftime(
            "%Y-%m-%d"
        )

        event_detail_rows.append(
            {
                "Ticker": ticker,
                "Earnings_Date": earnings_date.strftime("%Y-%m-%d"),
                "Event_Window_Start": event_window_rows["Date"].min().strftime("%Y-%m-%d"),
                "Event_Window_End": event_window_rows["Date"].max().strftime("%Y-%m-%d"),
                "Trading_Days_Marked": len(event_window_rows),
                "Source": earnings_row["Source"],
            }
        )

    event_detail_data = pd.DataFrame(event_detail_rows)

    return ticker_stock_data, event_detail_data


def create_all_event_windows(
    stock_data: pd.DataFrame,
    earnings_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create event windows for all selected tickers.
    """
    all_event_window_dataframes = []
    all_event_detail_dataframes = []

    for ticker in TICKERS:
        print(f"Creating event windows for {ticker}...")

        ticker_stock_data = stock_data[stock_data["Ticker"] == ticker].copy()
        ticker_earnings_data = earnings_data[earnings_data["Ticker"] == ticker].copy()

        if ticker_stock_data.empty:
            print(f"WARNING: No stock data found for {ticker}.")
            continue

        if ticker_earnings_data.empty:
            print(f"WARNING: No earnings dates found for {ticker}.")
            continue

        event_window_data, event_detail_data = create_event_windows_for_ticker(
            ticker_stock_data=ticker_stock_data,
            ticker_earnings_data=ticker_earnings_data,
            ticker=ticker,
        )

        event_rows = int(event_window_data["Event_Window"].sum())
        total_rows = len(event_window_data)

        print(
            f"{ticker}: {event_rows} event-window trading days marked "
            f"out of {total_rows} total trading days."
        )

        all_event_window_dataframes.append(event_window_data)
        all_event_detail_dataframes.append(event_detail_data)

    if not all_event_window_dataframes:
        raise ValueError("No event window data was created.")

    combined_event_window_data = pd.concat(
        all_event_window_dataframes,
        ignore_index=True,
    )

    combined_event_detail_data = pd.concat(
        all_event_detail_dataframes,
        ignore_index=True,
    )

    combined_event_window_data = combined_event_window_data.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    combined_event_detail_data = combined_event_detail_data.sort_values(
        ["Ticker", "Earnings_Date"]
    ).reset_index(drop=True)

    return combined_event_window_data, combined_event_detail_data


def create_event_window_summary(event_window_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table showing event-window coverage by ticker.
    """
    summary = (
        event_window_data.groupby("Ticker")
        .agg(
            Total_Trading_Days=("Date", "count"),
            Event_Window_Days=("Event_Window", "sum"),
        )
        .reset_index()
    )

    summary["Non_Event_Window_Days"] = (
        summary["Total_Trading_Days"] - summary["Event_Window_Days"]
    )

    summary["Event_Window_Rate"] = (
        summary["Event_Window_Days"] / summary["Total_Trading_Days"]
    ).round(4)

    summary["Expected_Event_Window_Days"] = EVENT_WINDOW_DAYS * 20

    summary["Coverage_Check"] = summary.apply(
        lambda row: "OK"
        if row["Event_Window_Days"] == row["Expected_Event_Window_Days"]
        else "CHECK",
        axis=1,
    )

    return summary


def save_outputs(
    event_window_data: pd.DataFrame,
    event_detail_data: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    """
    Save event-window datasets and summary outputs.
    """
    event_window_output_file = PROCESSED_DIR / "event_window_dataset.csv"
    event_detail_output_file = PROCESSED_DIR / "event_window_detail.csv"
    summary_output_file = OUTPUT_TABLES_DIR / "event_window_summary.csv"

    event_window_data_to_save = event_window_data.copy()
    event_window_data_to_save["Date"] = event_window_data_to_save["Date"].dt.strftime(
        "%Y-%m-%d"
    )

    event_window_data_to_save.to_csv(event_window_output_file, index=False)
    event_detail_data.to_csv(event_detail_output_file, index=False)
    summary.to_csv(summary_output_file, index=False)

    print(f"\nEvent window dataset saved to: {event_window_output_file}")
    print(f"Event window detail saved to: {event_detail_output_file}")
    print(f"Event window summary saved to: {summary_output_file}")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting event window construction")
    print(f"Event definition: {EVENT_WINDOW_DAYS} trading days before earnings dates")
    print("=" * 70)

    ensure_output_directories()

    stock_data = load_stock_data()
    earnings_data = load_earnings_dates()

    print(f"Loaded stock data rows: {len(stock_data)}")
    print(f"Loaded earnings date rows: {len(earnings_data)}")

    event_window_data, event_detail_data = create_all_event_windows(
        stock_data=stock_data,
        earnings_data=earnings_data,
    )

    summary = create_event_window_summary(event_window_data)

    print("\nEvent Window Summary:")
    print(summary)

    save_outputs(
        event_window_data=event_window_data,
        event_detail_data=event_detail_data,
        summary=summary,
    )

    print("\nEvent window construction completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()