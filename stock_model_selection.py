import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

from model_selection import (
    FEATURE_COLUMNS as TECHNICAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_technical_indicators,
    build_model_candidates,
    evaluate_model_with_time_series_cv,
    flatten_history_columns,
    sanitize_name,
)


DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_MARKET_SYMBOL = "SPY"
SENTIMENT_DIR = Path("sentiment_feature_outputs")
OUTPUT_DIR = Path("stock_model_selection_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SENTIMENT_FEATURE_COLUMNS = [
    "ticker_sentiment_7d_prior",
    "spy_sentiment_7d_prior",
    "ticker_article_count_7d_prior",
    "spy_article_count_7d_prior",
    "ticker_positive_share_7d_prior",
    "ticker_negative_share_7d_prior",
    "spy_positive_share_7d_prior",
    "spy_negative_share_7d_prior",
]

MODEL_FEATURE_COLUMNS = TECHNICAL_FEATURE_COLUMNS + SENTIMENT_FEATURE_COLUMNS


def parse_date(value):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Could not parse date: {value}")
    return parsed.normalize()


def resolve_date_range(start_date_text=None, end_date_text=None, years=DEFAULT_LOOKBACK_YEARS):
    end_date = parse_date(end_date_text) if end_date_text else pd.Timestamp.today().normalize()
    start_date = parse_date(start_date_text) if start_date_text else end_date - pd.DateOffset(years=years)

    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    return start_date, end_date


def build_sentiment_feature_path(ticker, market_symbol):
    return SENTIMENT_DIR / f"{ticker.lower()}_{market_symbol.lower()}_historical_sentiment_features.csv"


def download_ticker_history_for_range(ticker, start_date, end_date):
    history = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )

    if history.empty:
        return pd.DataFrame()

    history = history.reset_index()
    history = flatten_history_columns(history, ticker)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["date"] = history["Date"].dt.normalize()
    history["ticker"] = ticker

    history = add_technical_indicators(history)
    history = history.dropna(subset=TECHNICAL_FEATURE_COLUMNS + ["next_open"]).copy()
    return history


def load_sentiment_features(sentiment_path):
    if not sentiment_path.exists():
        raise FileNotFoundError(
            f"Sentiment feature file not found: {sentiment_path}. "
            "Run historical_sentiment_features.py first."
        )

    sentiment_df = pd.read_csv(sentiment_path)
    if sentiment_df.empty:
        raise ValueError(f"Sentiment feature file is empty: {sentiment_path}")

    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce").dt.normalize()
    sentiment_df = sentiment_df.dropna(subset=["date"]).copy()

    required_columns = ["date"] + SENTIMENT_FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in sentiment_df.columns]
    if missing_columns:
        raise ValueError(
            f"Sentiment feature file is missing required columns: {', '.join(missing_columns)}"
        )

    return sentiment_df[required_columns].copy()


def build_stock_dataset(ticker, start_date, end_date, market_symbol, sentiment_path=None):
    price_df = download_ticker_history_for_range(ticker, start_date, end_date)
    if price_df.empty:
        raise ValueError(f"No usable price history returned for ticker {ticker}.")

    final_sentiment_path = sentiment_path or build_sentiment_feature_path(ticker, market_symbol)
    sentiment_df = load_sentiment_features(final_sentiment_path)

    dataset = price_df.merge(sentiment_df, on="date", how="left")

    starting_rows = len(dataset)
    dataset = dataset.dropna(subset=SENTIMENT_FEATURE_COLUMNS).copy()
    dataset = dataset[
        (dataset["ticker_article_count_7d_prior"] > 0)
        & (dataset["spy_article_count_7d_prior"] > 0)
    ].copy()
    dataset = dataset.sort_values("date").reset_index(drop=True)

    dropped_rows = starting_rows - len(dataset)
    if dataset.empty:
        raise ValueError(
            "No training rows remain after merging sentiment features and dropping rows "
            "without prior ticker/SPY article coverage."
        )

    return dataset, final_sentiment_path, dropped_rows, starting_rows


def save_outputs(
    ticker,
    market_symbol,
    dataset,
    summary_df,
    fold_df,
    best_model_name,
    best_model,
    sentiment_path,
    dropped_rows,
    start_date,
    end_date,
):
    ticker_slug = sanitize_name(ticker)
    summary_path = OUTPUT_DIR / f"{ticker_slug}_summary.csv"
    folds_path = OUTPUT_DIR / f"{ticker_slug}_folds.csv"
    metadata_path = OUTPUT_DIR / f"{ticker_slug}_best_model.json"
    model_path = OUTPUT_DIR / f"{ticker_slug}_best_model.joblib"
    dataset_path = OUTPUT_DIR / f"{ticker_slug}_training_dataset.csv"

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    dataset.to_csv(dataset_path, index=False)

    joblib.dump(best_model, model_path)

    metadata = {
        "ticker": ticker,
        "market_symbol": market_symbol,
        "rows_used": int(len(dataset)),
        "rows_dropped_for_missing_sentiment": int(dropped_rows),
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "features": MODEL_FEATURE_COLUMNS,
        "technical_features": TECHNICAL_FEATURE_COLUMNS,
        "sentiment_features": SENTIMENT_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "sentiment_feature_path": str(sentiment_path),
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "folds_path": str(folds_path),
        "dataset_path": str(dataset_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return summary_path, folds_path, metadata_path, model_path, dataset_path


def select_model_for_stock(ticker, start_date, end_date, market_symbol=DEFAULT_MARKET_SYMBOL, sentiment_path=None):
    dataset, final_sentiment_path, dropped_rows, starting_rows = build_stock_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        market_symbol=market_symbol,
        sentiment_path=sentiment_path,
    )

    x = dataset[MODEL_FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    summary_rows = []
    fold_frames = []

    for model_name, pipeline in build_model_candidates().items():
        print(f"Evaluating {model_name}...")
        summary, fold_df = evaluate_model_with_time_series_cv(model_name, pipeline, x, y)
        summary_rows.append(summary)

        fold_df["model_name"] = model_name
        fold_frames.append(fold_df)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["f1_mean", "accuracy_mean", "roc_auc_mean"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    fold_df = pd.concat(fold_frames, ignore_index=True)
    best_model_name = summary_df.iloc[0]["model_name"]
    best_model = build_model_candidates()[best_model_name]
    best_model.fit(dataset[MODEL_FEATURE_COLUMNS], dataset[TARGET_COLUMN])

    paths = save_outputs(
        ticker=ticker,
        market_symbol=market_symbol,
        dataset=dataset,
        summary_df=summary_df,
        fold_df=fold_df,
        best_model_name=best_model_name,
        best_model=best_model,
        sentiment_path=final_sentiment_path,
        dropped_rows=dropped_rows,
        start_date=start_date,
        end_date=end_date,
    )

    return {
        "ticker": ticker,
        "market_symbol": market_symbol,
        "dataset_rows": len(dataset),
        "starting_rows": starting_rows,
        "dropped_rows": dropped_rows,
        "summary_df": summary_df,
        "best_model_name": best_model_name,
        "summary_path": paths[0],
        "folds_path": paths[1],
        "metadata_path": paths[2],
        "model_path": paths[3],
        "dataset_path": paths[4],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model selection for a single stock using technical and sentiment features."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol to train, e.g. AAPL")
    parser.add_argument(
        "--start-date",
        help="Optional start date in YYYY-MM-DD format. Defaults to 5 years before the end date.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_LOOKBACK_YEARS,
        help=f"Default lookback years when start date is not supplied. Default: {DEFAULT_LOOKBACK_YEARS}",
    )
    parser.add_argument(
        "--market-symbol",
        default=DEFAULT_MARKET_SYMBOL,
        help=f"Market benchmark used by the sentiment feature file. Default: {DEFAULT_MARKET_SYMBOL}",
    )
    parser.add_argument(
        "--sentiment-path",
        help="Optional explicit path to a historical sentiment feature CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper().strip()
    market_symbol = args.market_symbol.upper().strip()
    sentiment_path = Path(args.sentiment_path) if args.sentiment_path else None
    start_date, end_date = resolve_date_range(
        start_date_text=args.start_date,
        end_date_text=args.end_date,
        years=args.years,
    )

    print(f"Running stock model selection for {ticker}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Market symbol: {market_symbol}")
    if sentiment_path:
        print(f"Sentiment feature file: {sentiment_path}")
    else:
        print(f"Sentiment feature file: {build_sentiment_feature_path(ticker, market_symbol)}")

    result = select_model_for_stock(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        market_symbol=market_symbol,
        sentiment_path=sentiment_path,
    )

    print(f"\nStarting rows after technical feature build: {result['starting_rows']}")
    print(f"Rows dropped for missing prior article coverage: {result['dropped_rows']}")
    print(f"Rows used for model selection: {result['dataset_rows']}")
    print("\nModel ranking:")
    print(result["summary_df"].to_string(index=False))
    print(f"\nBest model: {result['best_model_name']}")
    print(f"Saved training dataset to: {result['dataset_path']}")
    print(f"Saved summary to: {result['summary_path']}")
    print(f"Saved fold metrics to: {result['folds_path']}")
    print(f"Saved best model metadata to: {result['metadata_path']}")
    print(f"Saved best fitted model to: {result['model_path']}")


if __name__ == "__main__":
    main()
