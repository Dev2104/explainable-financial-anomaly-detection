"""
create_earnings_csvs.py

Temporary helper script to create earnings announcement date CSV files
for the dissertation project.

These dates are used for creating 10-trading-day pre-earnings event windows.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EARNINGS_DIR = PROJECT_ROOT / "data" / "raw" / "earnings_dates"

EARNINGS_DATES = {
    "AAPL": [
        "2021-01-27", "2021-04-28", "2021-07-27", "2021-10-28",
        "2022-01-27", "2022-04-28", "2022-07-28", "2022-10-27",
        "2023-02-02", "2023-05-04", "2023-08-03", "2023-11-02",
        "2024-02-01", "2024-05-02", "2024-08-01", "2024-10-31",
        "2025-01-30", "2025-05-01", "2025-07-31", "2025-10-30",
    ],
    "MSFT": [
        "2021-01-26", "2021-04-27", "2021-07-27", "2021-10-26",
        "2022-01-25", "2022-04-26", "2022-07-26", "2022-10-25",
        "2023-01-24", "2023-04-25", "2023-07-25", "2023-10-24",
        "2024-01-30", "2024-04-25", "2024-07-30", "2024-10-30",
        "2025-01-29", "2025-04-30", "2025-07-30", "2025-10-29",
    ],
    "NVDA": [
        "2021-02-24", "2021-05-26", "2021-08-18", "2021-11-17",
        "2022-02-16", "2022-05-25", "2022-08-24", "2022-11-16",
        "2023-02-22", "2023-05-24", "2023-08-23", "2023-11-21",
        "2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20",
        "2025-02-26", "2025-05-28", "2025-08-27", "2025-11-19",
    ],
    "AMZN": [
        "2021-02-02", "2021-04-29", "2021-07-29", "2021-10-28",
        "2022-02-03", "2022-04-28", "2022-07-28", "2022-10-27",
        "2023-02-02", "2023-04-27", "2023-08-03", "2023-10-26",
        "2024-02-01", "2024-04-30", "2024-08-01", "2024-10-31",
        "2025-02-06", "2025-05-01", "2025-07-31", "2025-10-30",
    ],
    "TSLA": [
        "2021-01-27", "2021-04-26", "2021-07-26", "2021-10-20",
        "2022-01-26", "2022-04-20", "2022-07-20", "2022-10-19",
        "2023-01-25", "2023-04-19", "2023-07-19", "2023-10-18",
        "2024-01-24", "2024-04-23", "2024-07-23", "2024-10-23",
        "2025-01-29", "2025-04-22", "2025-07-23", "2025-10-22",
    ],
}


def create_earnings_files() -> None:
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)

    for ticker, dates in EARNINGS_DATES.items():
        data = pd.DataFrame(
            {
                "Ticker": [ticker] * len(dates),
                "Earnings_Date": dates,
                "Source": ["Investing.com"] * len(dates),
            }
        )

        output_file = EARNINGS_DIR / f"{ticker}_earnings_dates.csv"
        data.to_csv(output_file, index=False)

        print(f"Created {output_file.name} with {len(data)} earnings dates.")

    print("\nAll earnings date CSV files created successfully.")


if __name__ == "__main__":
    create_earnings_files()