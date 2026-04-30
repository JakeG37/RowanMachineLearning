import json
from pathlib import Path

import joblib

from stock_model_selection import (
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MARKET_SYMBOL,
    MODEL_FEATURE_COLUMNS,
    SENTIMENT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TECHNICAL_FEATURE_COLUMNS,
    build_stock_dataset,
    build_sentiment_feature_path,
    build_model_candidates,
    resolve_date_range,
    sanitize_name,
)


DEPLOY_DIR = Path("selected_stock_model_outputs")
DEPLOY_DIR.mkdir(exist_ok=True)

# Add more tickers here as you finalize model choices.
SELECTED_MODELS_BY_TICKER = {
    "AAPL": "svc_rbf",
}


def train_selected_model_for_ticker(
    ticker,
    model_name,
    start_date,
    end_date,
    market_symbol=DEFAULT_MARKET_SYMBOL,
    sentiment_path=None,
):
    candidates = build_model_candidates()
    if model_name not in candidates:
        raise ValueError(f"Model '{model_name}' is not defined in model_selection.py.")

    dataset, final_sentiment_path, dropped_rows, starting_rows = build_stock_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        market_symbol=market_symbol,
        sentiment_path=sentiment_path,
    )

    pipeline = candidates[model_name]
    pipeline.fit(dataset[MODEL_FEATURE_COLUMNS], dataset[TARGET_COLUMN])

    ticker_slug = sanitize_name(ticker)
    model_path = DEPLOY_DIR / f"{ticker_slug}_{model_name}.joblib"
    metadata_path = DEPLOY_DIR / f"{ticker_slug}_{model_name}.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "ticker": ticker,
        "selected_model_name": model_name,
        "market_symbol": market_symbol,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "rows_used": int(len(dataset)),
        "starting_rows_after_technical_features": int(starting_rows),
        "rows_dropped_for_missing_sentiment": int(dropped_rows),
        "features": MODEL_FEATURE_COLUMNS,
        "technical_features": TECHNICAL_FEATURE_COLUMNS,
        "sentiment_features": SENTIMENT_FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "sentiment_feature_path": str(final_sentiment_path),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return model_path, metadata_path, metadata


def load_selected_model_for_ticker(ticker):
    ticker = ticker.upper().strip()
    model_name = SELECTED_MODELS_BY_TICKER.get(ticker)
    if not model_name:
        raise ValueError(f"No selected model configured for ticker {ticker}.")

    ticker_slug = sanitize_name(ticker)
    model_path = DEPLOY_DIR / f"{ticker_slug}_{model_name}.joblib"
    metadata_path = DEPLOY_DIR / f"{ticker_slug}_{model_name}.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model file not found for {ticker}. Run selected_stock_models.py first."
        )

    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, metadata


def main():
    start_date, end_date = resolve_date_range(years=DEFAULT_LOOKBACK_YEARS)

    for ticker, model_name in SELECTED_MODELS_BY_TICKER.items():
        sentiment_path = build_sentiment_feature_path(ticker, DEFAULT_MARKET_SYMBOL)

        print(f"\nTraining selected model for {ticker}")
        print(f"Chosen model: {model_name}")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        print(f"Sentiment feature file: {sentiment_path}")

        model_path, metadata_path, metadata = train_selected_model_for_ticker(
            ticker=ticker,
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            market_symbol=DEFAULT_MARKET_SYMBOL,
            sentiment_path=sentiment_path,
        )

        print(f"Rows used: {metadata['rows_used']}")
        print(f"Rows dropped for missing sentiment: {metadata['rows_dropped_for_missing_sentiment']}")
        print(f"Saved selected model to: {model_path}")
        print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
