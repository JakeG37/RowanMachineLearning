import os

import pandas as pd
import yfinance as yf

from historical_sentiment_features import build_daily_sentiment_frame, build_scored_news_dataset
from model_selection import add_technical_indicators, flatten_history_columns
from selected_stock_models import load_selected_model_for_ticker
from stock_model_selection import (
    DEFAULT_MARKET_SYMBOL,
    MODEL_FEATURE_COLUMNS,
    TECHNICAL_FEATURE_COLUMNS,
)


def download_recent_price_history(ticker, years=2):
    history = yf.download(
        ticker,
        period=f"{years}y",
        progress=False,
        auto_adjust=False,
    )

    if history.empty:
        raise ValueError(f"No recent market history was returned for ticker {ticker}.")

    history = history.reset_index()
    history = flatten_history_columns(history, ticker)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["date"] = history["Date"].dt.normalize()
    history["ticker"] = ticker
    history = add_technical_indicators(history)
    return history


def build_live_sentiment_feature_row(
    ticker,
    latest_trading_date,
    market_symbol=DEFAULT_MARKET_SYMBOL,
    api_key=None,
    max_news_articles=None,
):
    final_api_key = api_key or os.getenv("FINNHUB_API_KEY")
    if not final_api_key:
        raise ValueError("Set the FINNHUB_API_KEY environment variable before running live prediction.")

    sentiment_start_date = latest_trading_date - pd.Timedelta(days=14)
    sentiment_end_date = latest_trading_date

    ticker_articles = build_scored_news_dataset(
        ticker,
        final_api_key,
        sentiment_start_date,
        sentiment_end_date,
        max_articles=max_news_articles,
    )
    market_articles = build_scored_news_dataset(
        market_symbol,
        final_api_key,
        sentiment_start_date,
        sentiment_end_date,
        max_articles=max_news_articles,
    )

    ticker_daily = build_daily_sentiment_frame(
        ticker_articles,
        start_date=sentiment_start_date,
        end_date=sentiment_end_date,
        prefix="ticker",
    )
    market_daily = build_daily_sentiment_frame(
        market_articles,
        start_date=sentiment_start_date,
        end_date=sentiment_end_date,
        prefix="spy",
    )

    ticker_row = ticker_daily.loc[ticker_daily["date"] == latest_trading_date]
    market_row = market_daily.loc[market_daily["date"] == latest_trading_date]

    if ticker_row.empty or market_row.empty:
        latest_date_text = pd.Timestamp(latest_trading_date).date().isoformat()
        raise ValueError(
            f"Prediction unavailable: could not align the latest trading date {latest_date_text} "
            f"with the live sentiment feature tables."
        )

    merged_row = pd.concat(
        [
            ticker_row.reset_index(drop=True),
            market_row.drop(columns=["date"]).reset_index(drop=True),
        ],
        axis=1,
    )
    return merged_row.iloc[0], ticker_articles, market_articles


def build_live_prediction_feature_row(
    ticker,
    market_symbol=DEFAULT_MARKET_SYMBOL,
    years=2,
    api_key=None,
    max_news_articles=None,
):
    history = download_recent_price_history(ticker, years=years)
    history = history.dropna(subset=TECHNICAL_FEATURE_COLUMNS).copy()

    if history.empty:
        raise ValueError(f"Not enough price history to compute live features for ticker {ticker}.")

    latest_row = history.sort_values("date").iloc[-1]
    latest_trading_date = pd.Timestamp(latest_row["date"]).normalize()

    sentiment_row, ticker_articles, market_articles = build_live_sentiment_feature_row(
        ticker=ticker,
        latest_trading_date=latest_trading_date,
        market_symbol=market_symbol,
        api_key=api_key,
        max_news_articles=max_news_articles,
    )

    combined_features = latest_row.to_dict()
    combined_features.update(sentiment_row.to_dict())
    feature_frame = pd.DataFrame([combined_features])

    missing_columns = [column for column in MODEL_FEATURE_COLUMNS if column not in feature_frame.columns]
    if missing_columns:
        raise ValueError(
            f"Live prediction feature row is missing required columns: {', '.join(missing_columns)}"
        )

    return feature_frame[MODEL_FEATURE_COLUMNS], latest_row, sentiment_row, ticker_articles, market_articles


def predict_with_live_stock_model(ticker, api_key=None, max_news_articles=None):
    model, metadata = load_selected_model_for_ticker(ticker)
    market_symbol = metadata.get("market_symbol", DEFAULT_MARKET_SYMBOL)

    feature_frame, latest_row, sentiment_row, ticker_articles, market_articles = (
        build_live_prediction_feature_row(
            ticker=ticker,
            market_symbol=market_symbol,
            api_key=api_key,
            max_news_articles=max_news_articles,
        )
    )

    prediction = int(model.predict(feature_frame)[0])
    probability_up = None
    if hasattr(model, "predict_proba"):
        probability_up = float(model.predict_proba(feature_frame)[0][1])

    return {
        "prediction": prediction,
        "prediction_label": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "latest_close": float(latest_row["Close"]),
        "latest_date": latest_row["date"],
        "market_symbol": market_symbol,
        "model_metadata": metadata,
        "features": feature_frame.iloc[0].to_dict(),
        "ticker_sentiment_7d_prior": float(sentiment_row["ticker_sentiment_7d_prior"]),
        "spy_sentiment_7d_prior": float(sentiment_row["spy_sentiment_7d_prior"]),
        "ticker_article_count_7d_prior": int(sentiment_row["ticker_article_count_7d_prior"]),
        "spy_article_count_7d_prior": int(sentiment_row["spy_article_count_7d_prior"]),
        "ticker_articles_used": int(len(ticker_articles)),
        "market_articles_used": int(len(market_articles)),
    }


def main():
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("Set the FINNHUB_API_KEY environment variable before running this script.")

    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
    result = predict_with_live_stock_model(ticker, api_key=api_key)

    print(f"\nTicker: {ticker}")
    print(f"Prediction: {result['prediction_label']}")
    print(f"Probability up: {result['probability_up']}")
    print(f"Latest trading date: {result['latest_date']}")
    print(f"Ticker prior 7d sentiment: {result['ticker_sentiment_7d_prior']}")
    print(f"{result['market_symbol']} prior 7d sentiment: {result['spy_sentiment_7d_prior']}")


if __name__ == "__main__":
    main()
