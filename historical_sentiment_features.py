import argparse
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


SENTIMENT_TO_SCORE = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0,
}

DEFAULT_LOOKBACK_YEARS = 5
OUTPUT_DIR = Path("sentiment_feature_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
DEFAULT_NEWS_REQUEST_DELAY_SECONDS = 1.5
DEFAULT_SENTIMENT_DELAY_SECONDS = 0.15
DEFAULT_MAX_RETRIES = 5

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


def download_trading_dates(ticker, start_date, end_date):
    history = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )

    if history.empty:
        raise ValueError(f"No price history returned for ticker {ticker} in the requested date range.")

    history = history.reset_index()
    history = flatten_history_columns(history, ticker)
    history["Date"] = pd.to_datetime(history["Date"], errors="coerce")
    history["date"] = history["Date"].dt.normalize()
    return history[["date"]].dropna().drop_duplicates().sort_values("date").reset_index(drop=True)


def fetch_company_news(
    symbol,
    api_key,
    start_date,
    end_date,
    chunk_days=30,
    pause_seconds=DEFAULT_NEWS_REQUEST_DELAY_SECONDS,
    max_retries=DEFAULT_MAX_RETRIES,
):
    all_articles = []
    window_start = start_date.normalize()

    while window_start <= end_date:
        window_end = min(window_start + pd.Timedelta(days=chunk_days - 1), end_date)
        url = (
            f"https://finnhub.io/api/v1/company-news"
            f"?symbol={symbol}"
            f"&from={window_start.strftime('%Y-%m-%d')}"
            f"&to={window_end.strftime('%Y-%m-%d')}"
            f"&token={api_key}"
        )

        payload = None
        for attempt in range(max_retries):
            response = requests.get(url, timeout=30)

            if response.status_code != 429:
                response.raise_for_status()
                payload = response.json()
                break

            retry_delay = pause_seconds * (2 ** attempt)
            print(
                f"Rate limited while fetching {symbol} news for "
                f"{window_start.date()} to {window_end.date()}. "
                f"Sleeping {retry_delay:.1f}s before retry {attempt + 1}/{max_retries}."
            )
            time.sleep(retry_delay)

        if payload is None:
            raise RuntimeError(
                f"Exceeded retry limit while fetching news for {symbol} "
                f"from {window_start.date()} to {window_end.date()}."
            )

        if isinstance(payload, list):
            all_articles.extend(payload)

        print(
            f"Fetched {len(payload) if isinstance(payload, list) else 0} raw articles "
            f"for {symbol} from {window_start.date()} to {window_end.date()}"
        )

        window_start = window_end + pd.Timedelta(days=1)
        time.sleep(pause_seconds)

    return all_articles


def parse_news_articles(symbol, raw_articles):
    parsed_rows = []

    for article in raw_articles:
        timestamp_value = article.get("datetime")
        if not timestamp_value:
            continue

        published = pd.to_datetime(int(timestamp_value), unit="s", errors="coerce")
        if pd.isna(published):
            continue

        title = (article.get("headline") or "").strip()
        summary = (article.get("summary") or "").strip()
        url = article.get("url")

        parsed_rows.append(
            {
                "symbol": symbol,
                "title": title,
                "summary": summary,
                "source": article.get("source"),
                "url": url,
                "published": published,
                "published_date": published.normalize(),
            }
        )

    article_df = pd.DataFrame(parsed_rows)
    if article_df.empty:
        return article_df

    article_df = article_df.drop_duplicates(
        subset=["published_date", "title", "url"],
        keep="first",
    ).reset_index(drop=True)
    article_df = article_df.sort_values("published", ascending=False).reset_index(drop=True)
    return article_df


def score_article_frame(article_df, pause_seconds=DEFAULT_SENTIMENT_DELAY_SECONDS):
    if article_df.empty:
        return article_df

    scored_rows = []
    for article in article_df.itertuples(index=False):
        text_for_scoring = article.summary or article.title
        label, confidence, score = score_text_sentiment(text_for_scoring)

        scored_rows.append(
            {
                "symbol": article.symbol,
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


def build_scored_news_dataset(
    symbol,
    api_key,
    start_date,
    end_date,
    news_request_delay=DEFAULT_NEWS_REQUEST_DELAY_SECONDS,
    sentiment_delay=DEFAULT_SENTIMENT_DELAY_SECONDS,
    max_articles=None,
):
    raw_articles = fetch_company_news(
        symbol,
        api_key,
        start_date,
        end_date,
        pause_seconds=news_request_delay,
    )
    parsed_articles = parse_news_articles(symbol, raw_articles)
    if max_articles is not None:
        parsed_articles = parsed_articles.head(max_articles).copy()
    print(f"Scoring {len(parsed_articles)} unique articles for {symbol}")
    return score_article_frame(parsed_articles, pause_seconds=sentiment_delay)


def build_daily_sentiment_frame(article_df, start_date, end_date, prefix):
    calendar_df = pd.DataFrame(
        {"date": pd.date_range(start=start_date, end=end_date, freq="D").normalize()}
    )

    base_columns = [
        f"{prefix}_article_count",
        f"{prefix}_avg_sentiment",
        f"{prefix}_weighted_sentiment",
        f"{prefix}_positive_count",
        f"{prefix}_negative_count",
        f"{prefix}_neutral_count",
        f"{prefix}_sentiment_7d_prior",
        f"{prefix}_article_count_7d_prior",
    ]

    if article_df.empty:
        for column in base_columns:
            calendar_df[column] = 0.5 if "sentiment" in column else 0
        calendar_df[f"{prefix}_positive_share_7d_prior"] = 0.0
        calendar_df[f"{prefix}_negative_share_7d_prior"] = 0.0
        return calendar_df

    working_df = article_df.copy()
    working_df["published_date"] = pd.to_datetime(working_df["published_date"]).dt.normalize()
    working_df["sentiment_score"] = working_df["sentiment_score"].fillna(0.5)
    working_df["sentiment_confidence"] = working_df["sentiment_confidence"].fillna(0.0)
    working_df["weighted_component"] = working_df["sentiment_score"] * working_df["sentiment_confidence"]

    daily_df = (
        working_df.groupby("published_date", as_index=False)
        .agg(
            article_count=("title", "size"),
            avg_sentiment=("sentiment_score", "mean"),
            weight_sum=("sentiment_confidence", "sum"),
            weighted_component_sum=("weighted_component", "sum"),
            positive_count=("sentiment_label", lambda values: (values.str.lower() == "positive").sum()),
            negative_count=("sentiment_label", lambda values: (values.str.lower() == "negative").sum()),
            neutral_count=("sentiment_label", lambda values: (values.str.lower() == "neutral").sum()),
        )
        .rename(columns={"published_date": "date"})
    )

    daily_df["weighted_sentiment"] = daily_df["weighted_component_sum"] / daily_df["weight_sum"]
    daily_df["weighted_sentiment"] = daily_df["weighted_sentiment"].fillna(daily_df["avg_sentiment"])
    daily_df = daily_df.drop(columns=["weight_sum", "weighted_component_sum"])

    rename_map = {
        "article_count": f"{prefix}_article_count",
        "avg_sentiment": f"{prefix}_avg_sentiment",
        "weighted_sentiment": f"{prefix}_weighted_sentiment",
        "positive_count": f"{prefix}_positive_count",
        "negative_count": f"{prefix}_negative_count",
        "neutral_count": f"{prefix}_neutral_count",
    }
    daily_df = daily_df.rename(columns=rename_map)

    calendar_df = calendar_df.merge(daily_df, on="date", how="left")

    count_columns = [column for column in calendar_df.columns if column.endswith("_count")]
    sentiment_columns = [column for column in calendar_df.columns if "sentiment" in column]

    for column in count_columns:
        calendar_df[column] = calendar_df[column].fillna(0)

    for column in sentiment_columns:
        if column not in {f"{prefix}_sentiment_7d_prior"}:
            calendar_df[column] = calendar_df[column].fillna(0.5)

    count_col = f"{prefix}_article_count"
    score_col = f"{prefix}_weighted_sentiment"
    positive_col = f"{prefix}_positive_count"
    negative_col = f"{prefix}_negative_count"

    prior_weighted_sum = (calendar_df[score_col] * calendar_df[count_col]).shift(1).rolling(
        window=7,
        min_periods=1,
    ).sum()
    prior_article_count = calendar_df[count_col].shift(1).rolling(window=7, min_periods=1).sum()
    prior_positive_count = calendar_df[positive_col].shift(1).rolling(window=7, min_periods=1).sum()
    prior_negative_count = calendar_df[negative_col].shift(1).rolling(window=7, min_periods=1).sum()

    calendar_df[f"{prefix}_sentiment_7d_prior"] = prior_weighted_sum / prior_article_count
    calendar_df[f"{prefix}_sentiment_7d_prior"] = calendar_df[f"{prefix}_sentiment_7d_prior"].fillna(0.5)
    calendar_df[f"{prefix}_article_count_7d_prior"] = prior_article_count.fillna(0).astype(int)
    calendar_df[f"{prefix}_positive_share_7d_prior"] = (
        prior_positive_count / prior_article_count
    ).fillna(0.0)
    calendar_df[f"{prefix}_negative_share_7d_prior"] = (
        prior_negative_count / prior_article_count
    ).fillna(0.0)

    for column in count_columns:
        calendar_df[column] = calendar_df[column].astype(int)

    return calendar_df


def build_sentiment_feature_tables(
    ticker,
    start_date,
    end_date,
    market_symbol,
    api_key,
    news_request_delay=DEFAULT_NEWS_REQUEST_DELAY_SECONDS,
    sentiment_delay=DEFAULT_SENTIMENT_DELAY_SECONDS,
):
    trading_dates = download_trading_dates(ticker, start_date, end_date)

    ticker_articles = build_scored_news_dataset(
        ticker,
        api_key,
        start_date,
        end_date,
        news_request_delay=news_request_delay,
        sentiment_delay=sentiment_delay,
    )
    market_articles = build_scored_news_dataset(
        market_symbol,
        api_key,
        start_date,
        end_date,
        news_request_delay=news_request_delay,
        sentiment_delay=sentiment_delay,
    )

    ticker_daily = build_daily_sentiment_frame(ticker_articles, start_date, end_date, prefix="ticker")
    market_daily = build_daily_sentiment_frame(market_articles, start_date, end_date, prefix="spy")

    feature_df = trading_dates.merge(
        ticker_daily[
            [
                "date",
                "ticker_sentiment_7d_prior",
                "ticker_article_count_7d_prior",
                "ticker_positive_share_7d_prior",
                "ticker_negative_share_7d_prior",
            ]
        ],
        on="date",
        how="left",
    )
    feature_df = feature_df.merge(
        market_daily[
            [
                "date",
                "spy_sentiment_7d_prior",
                "spy_article_count_7d_prior",
                "spy_positive_share_7d_prior",
                "spy_negative_share_7d_prior",
            ]
        ],
        on="date",
        how="left",
    )

    feature_df["ticker"] = ticker
    feature_df["market_symbol"] = market_symbol
    feature_df = feature_df.sort_values("date").reset_index(drop=True)

    return feature_df, ticker_articles, market_articles, ticker_daily, market_daily


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build historical ticker and SPY sentiment features for per-stock models."
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol to analyze, e.g. AAPL")
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
        default="SPY",
        help="Market benchmark symbol for the broad-market sentiment feature. Default: SPY",
    )
    parser.add_argument(
        "--news-delay",
        type=float,
        default=DEFAULT_NEWS_REQUEST_DELAY_SECONDS,
        help=(
            "Delay in seconds between Finnhub news requests. "
            f"Default: {DEFAULT_NEWS_REQUEST_DELAY_SECONDS}"
        ),
    )
    parser.add_argument(
        "--sentiment-delay",
        type=float,
        default=DEFAULT_SENTIMENT_DELAY_SECONDS,
        help=(
            "Delay in seconds between FinBERT article scoring calls. "
            f"Default: {DEFAULT_SENTIMENT_DELAY_SECONDS}"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper().strip()
    market_symbol = args.market_symbol.upper().strip()

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("Set the FINNHUB_API_KEY environment variable before running this script.")

    start_date, end_date = resolve_date_range(
        start_date_text=args.start_date,
        end_date_text=args.end_date,
        years=args.years,
    )

    print(f"Building historical sentiment features for {ticker}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Market benchmark: {market_symbol}")
    print(f"News request delay: {args.news_delay:.2f}s")
    print(f"Sentiment scoring delay: {args.sentiment_delay:.2f}s")

    feature_df, ticker_articles, market_articles, ticker_daily, market_daily = (
        build_sentiment_feature_tables(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            market_symbol=market_symbol,
            api_key=api_key,
            news_request_delay=args.news_delay,
            sentiment_delay=args.sentiment_delay,
        )
    )

    feature_path = OUTPUT_DIR / f"{ticker.lower()}_{market_symbol.lower()}_historical_sentiment_features.csv"
    ticker_articles_path = OUTPUT_DIR / f"{ticker.lower()}_historical_articles_scored.csv"
    market_articles_path = OUTPUT_DIR / f"{market_symbol.lower()}_historical_articles_scored.csv"
    ticker_daily_path = OUTPUT_DIR / f"{ticker.lower()}_daily_sentiment.csv"
    market_daily_path = OUTPUT_DIR / f"{market_symbol.lower()}_daily_sentiment.csv"

    feature_df.to_csv(feature_path, index=False)
    ticker_articles.to_csv(ticker_articles_path, index=False)
    market_articles.to_csv(market_articles_path, index=False)
    ticker_daily.to_csv(ticker_daily_path, index=False)
    market_daily.to_csv(market_daily_path, index=False)

    print(f"Saved merge-ready feature file to: {feature_path}")
    print(f"Saved scored {ticker} articles to: {ticker_articles_path}")
    print(f"Saved scored {market_symbol} articles to: {market_articles_path}")
    print(f"Saved daily {ticker} sentiment history to: {ticker_daily_path}")
    print(f"Saved daily {market_symbol} sentiment history to: {market_daily_path}")
    print("\nFeature sample:")
    print(feature_df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
