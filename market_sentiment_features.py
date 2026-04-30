import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


SENTIMENT_TO_SCORE = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0,
}

OUTPUT_DIR = Path("sentiment_feature_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

_finbert = None


def get_finbert_pipeline():
    global _finbert

    if _finbert is None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required to score sentiment. Activate MLvenv or install transformers."
            ) from exc

        hf_token = os.getenv("HF_TOKEN")
        _finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            token=hf_token if hf_token else None,
        )
        print("FinBERT loaded successfully")

    return _finbert


def score_text_sentiment(text):
    if not text:
        return None, None, None

    result = get_finbert_pipeline()(text[:512])[0]
    label = result["label"]
    confidence = float(result["score"])
    score = SENTIMENT_TO_SCORE.get(label.lower(), 0.5)
    return label, confidence, score


def flatten_history_columns(history, ticker):
    if isinstance(history.columns, pd.MultiIndex):
        history.columns = [
            column[0] if isinstance(column, tuple) and column[1] == ticker else column[0]
            for column in history.columns
        ]
    return history


def compute_return_streak(returns):
    direction = pd.Series(0, index=returns.index, dtype="float64")
    direction.loc[returns > 0] = 1.0
    direction.loc[returns < 0] = -1.0

    streak_change = direction.ne(direction.shift(fill_value=0))
    streak_groups = streak_change.cumsum()
    streak_lengths = direction.groupby(streak_groups).cumcount() + 1
    return streak_lengths.where(direction != 0, 0).astype(float) * direction


def download_price_history(ticker, start_date, end_date):
    history = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )

    if history.empty:
        return pd.DataFrame()

    history = history.reset_index()
    history = flatten_history_columns(history, ticker)
    history["Date"] = pd.to_datetime(history["Date"])
    history["date"] = history["Date"].dt.normalize()
    history["ticker"] = ticker
    return history


def build_market_price_features(market_symbol, start_date, end_date):
    history = download_price_history(market_symbol, start_date, end_date)
    if history.empty:
        return pd.DataFrame()

    history["market_daily_return"] = history["Close"].pct_change()
    history["market_return_streak"] = compute_return_streak(history["market_daily_return"])
    history["market_volatility_5"] = history["market_daily_return"].rolling(window=5).std()
    history["market_ma_5"] = history["Close"].rolling(window=5).mean()
    history["market_ma_ratio_5"] = history["Close"] / history["market_ma_5"]

    return history[
        [
            "date",
            "market_daily_return",
            "market_return_streak",
            "market_volatility_5",
            "market_ma_ratio_5",
        ]
    ].copy()


def fetch_company_news(symbol, api_key, start_date, end_date, chunk_days=30, pause_seconds=0.25):
    all_articles = []
    window_start = pd.Timestamp(start_date).normalize()
    final_date = pd.Timestamp(end_date).normalize()

    while window_start <= final_date:
        window_end = min(window_start + pd.Timedelta(days=chunk_days - 1), final_date)
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={symbol}"
            f"&from={window_start.strftime('%Y-%m-%d')}"
            f"&to={window_end.strftime('%Y-%m-%d')}"
            f"&token={api_key}"
        )

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            all_articles.extend(payload)

        print(
            f"Fetched {len(payload) if isinstance(payload, list) else 0} raw articles "
            f"for {symbol} from {window_start.date()} to {window_end.date()}"
        )

        window_start = window_end + pd.Timedelta(days=1)
        time.sleep(pause_seconds)

    return all_articles


def parse_news_articles(raw_articles):
    parsed = []

    for article in raw_articles:
        published_ts = article.get("datetime")
        if not published_ts:
            continue

        published = pd.to_datetime(int(published_ts), unit="s", errors="coerce")
        if pd.isna(published):
            continue

        title = (article.get("headline") or "").strip()
        summary = (article.get("summary") or "").strip()
        url = article.get("url")

        parsed.append(
            {
                "title": title,
                "summary": summary,
                "source": article.get("source"),
                "url": url,
                "published": published,
            }
        )

    articles_df = pd.DataFrame(parsed)
    if articles_df.empty:
        return articles_df

    articles_df["published_date"] = pd.to_datetime(articles_df["published"]).dt.normalize()
    articles_df = articles_df.drop_duplicates(
        subset=["published_date", "title", "url"],
        keep="first",
    ).reset_index(drop=True)
    return articles_df


def score_article_frame(symbol, article_df, pause_seconds=0.05):
    if article_df.empty:
        return article_df

    scored_rows = []
    for article in article_df.itertuples(index=False):
        text_for_scoring = article.summary or article.title
        label, confidence, score = score_text_sentiment(text_for_scoring)

        scored_rows.append(
            {
                "symbol": symbol,
                "title": article.title,
                "summary": article.summary,
                "source": article.source,
                "url": article.url,
                "published": article.published,
                "published_date": article.published_date,
                "sentiment_label": label,
                "sentiment_confidence": confidence,
                "sentiment_score": score,
            }
        )
        time.sleep(pause_seconds)

    return pd.DataFrame(scored_rows)


def build_news_dataset(symbol, api_key, start_date, end_date):
    raw_articles = fetch_company_news(symbol, api_key, start_date, end_date)
    parsed_articles = parse_news_articles(raw_articles)
    print(f"Scoring {len(parsed_articles)} unique articles for {symbol}")
    return score_article_frame(symbol, parsed_articles)


def aggregate_daily_sentiment(article_df, prefix):
    base_columns = [
        f"{prefix}_article_count",
        f"{prefix}_average_sentiment",
        f"{prefix}_weighted_sentiment",
        f"{prefix}_positive_count",
        f"{prefix}_negative_count",
        f"{prefix}_neutral_count",
    ]

    if article_df.empty:
        return pd.DataFrame(columns=["date"] + base_columns)

    working_df = article_df.copy()
    working_df["date"] = pd.to_datetime(working_df["published_date"]).dt.normalize()
    working_df["sentiment_confidence"] = working_df["sentiment_confidence"].fillna(0.0)
    working_df["sentiment_score"] = working_df["sentiment_score"].fillna(0.5)
    working_df["weighted_score_component"] = (
        working_df["sentiment_score"] * working_df["sentiment_confidence"]
    )

    grouped = (
        working_df.groupby("date", as_index=False)
        .agg(
            article_count=("title", "size"),
            average_sentiment=("sentiment_score", "mean"),
            weighted_score_sum=("weighted_score_component", "sum"),
            weight_sum=("sentiment_confidence", "sum"),
            positive_count=("sentiment_label", lambda values: (values.str.lower() == "positive").sum()),
            negative_count=("sentiment_label", lambda values: (values.str.lower() == "negative").sum()),
            neutral_count=("sentiment_label", lambda values: (values.str.lower() == "neutral").sum()),
        )
    )

    grouped["weighted_sentiment"] = grouped["weighted_score_sum"] / grouped["weight_sum"]
    grouped["weighted_sentiment"] = grouped["weighted_sentiment"].fillna(grouped["average_sentiment"])
    grouped = grouped.drop(columns=["weighted_score_sum", "weight_sum"])

    rename_map = {
        "article_count": f"{prefix}_article_count",
        "average_sentiment": f"{prefix}_average_sentiment",
        "weighted_sentiment": f"{prefix}_weighted_sentiment",
        "positive_count": f"{prefix}_positive_count",
        "negative_count": f"{prefix}_negative_count",
        "neutral_count": f"{prefix}_neutral_count",
    }
    return grouped.rename(columns=rename_map)


def add_sentiment_rollups(feature_df, prefix, windows=(3, 7)):
    score_col = f"{prefix}_weighted_sentiment"
    count_col = f"{prefix}_article_count"

    feature_df[count_col] = feature_df[count_col].fillna(0)
    feature_df[score_col] = feature_df[score_col].fillna(0.5)

    weighted_sum_col = f"{prefix}_weighted_sum"
    feature_df[weighted_sum_col] = feature_df[score_col] * feature_df[count_col]

    for window in windows:
        rolling_count = feature_df[count_col].rolling(window=window, min_periods=1).sum()
        rolling_score_sum = feature_df[weighted_sum_col].rolling(window=window, min_periods=1).sum()

        roll_col = f"{prefix}_sentiment_{window}d"
        feature_df[roll_col] = rolling_score_sum / rolling_count
        feature_df[roll_col] = feature_df[roll_col].fillna(0.5)

        positive_col = f"{prefix}_positive_count"
        negative_col = f"{prefix}_negative_count"

        if positive_col in feature_df.columns:
            positive_share_col = f"{prefix}_positive_share_{window}d"
            positive_sum = feature_df[positive_col].fillna(0).rolling(window=window, min_periods=1).sum()
            feature_df[positive_share_col] = (positive_sum / rolling_count).fillna(0.0)

        if negative_col in feature_df.columns:
            negative_share_col = f"{prefix}_negative_share_{window}d"
            negative_sum = feature_df[negative_col].fillna(0).rolling(window=window, min_periods=1).sum()
            feature_df[negative_share_col] = (negative_sum / rolling_count).fillna(0.0)

    return feature_df.drop(columns=[weighted_sum_col])


def build_sentiment_feature_frame(ticker, market_symbol, years, api_key):
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)

    ticker_history = download_price_history(ticker, start_date, end_date)
    if ticker_history.empty:
        raise ValueError(f"No price history returned for ticker {ticker}.")

    feature_df = ticker_history[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)

    market_price_features = build_market_price_features(market_symbol, start_date, end_date)
    if market_price_features.empty:
        raise ValueError(f"No market price history returned for benchmark {market_symbol}.")

    ticker_news = build_news_dataset(ticker, api_key, start_date, end_date)
    market_news = build_news_dataset(market_symbol, api_key, start_date, end_date)

    ticker_daily = aggregate_daily_sentiment(ticker_news, prefix="ticker_news")
    market_daily = aggregate_daily_sentiment(market_news, prefix="market_news")

    feature_df = feature_df.merge(market_price_features, on="date", how="left")
    feature_df = feature_df.merge(ticker_daily, on="date", how="left")
    feature_df = feature_df.merge(market_daily, on="date", how="left")

    feature_df = add_sentiment_rollups(feature_df, prefix="ticker_news")
    feature_df = add_sentiment_rollups(feature_df, prefix="market_news")

    count_columns = [column for column in feature_df.columns if column.endswith("_count")]
    share_columns = [column for column in feature_df.columns if "_share_" in column]
    sentiment_columns = [
        column
        for column in feature_df.columns
        if "sentiment" in column and column not in {"market_daily_return"}
    ]

    for column in count_columns:
        feature_df[column] = feature_df[column].fillna(0).astype(int)

    for column in share_columns:
        feature_df[column] = feature_df[column].fillna(0.0)

    for column in sentiment_columns:
        feature_df[column] = feature_df[column].fillna(0.5)

    feature_df["ticker"] = ticker
    feature_df["market_symbol"] = market_symbol

    return (
        feature_df.sort_values("date").reset_index(drop=True),
        ticker_news.sort_values("published").reset_index(drop=True),
        market_news.sort_values("published").reset_index(drop=True),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build daily market and ticker sentiment features for later model joins."
    )
    parser.add_argument("--ticker", required=True, help="Stock ticker to build features for, e.g. AAPL")
    parser.add_argument(
        "--market-symbol",
        default="SPY",
        help="Market benchmark symbol used as the broad market proxy. Default: SPY",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="How many years of trading dates and news to include. Default: 5",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper().strip()
    market_symbol = args.market_symbol.upper().strip()

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("Set the FINNHUB_API_KEY environment variable before running this script.")

    print(f"Building sentiment features for {ticker}")
    print(f"Market proxy: {market_symbol}")
    print(f"Lookback years: {args.years}")

    feature_df, ticker_news, market_news = build_sentiment_feature_frame(
        ticker=ticker,
        market_symbol=market_symbol,
        years=args.years,
        api_key=api_key,
    )

    feature_path = OUTPUT_DIR / f"{ticker.lower()}_{market_symbol.lower()}_sentiment_features.csv"
    ticker_news_path = OUTPUT_DIR / f"{ticker.lower()}_news_scored.csv"
    market_news_path = OUTPUT_DIR / f"{market_symbol.lower()}_news_scored.csv"

    feature_df.to_csv(feature_path, index=False)
    ticker_news.to_csv(ticker_news_path, index=False)
    market_news.to_csv(market_news_path, index=False)

    print(f"Saved feature frame to: {feature_path}")
    print(f"Saved ticker news scores to: {ticker_news_path}")
    print(f"Saved market news scores to: {market_news_path}")
    print("\nFeature sample:")
    print(feature_df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
