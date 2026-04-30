import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf
from edgar_filings import build_edgar_dataset
from newspaper import Article
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


SENTIMENT_TO_SCORE = {
    "positive": 1.0,
    "neutral": 0.5,
    "negative": 0.0,
}


finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
)
print("FinBERT loaded successfully")

summarizer = None
summarizer_tokenizer = None


def clean_text(text):
    if not text:
        return ""

    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def safe_string(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def get_stock_news(ticker, api_key, days_back=7):
    today = datetime.today()
    past = today - timedelta(days=days_back)

    from_date = past.strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")

    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}"
        f"&from={from_date}"
        f"&to={to_date}"
        f"&token={api_key}"
    )

    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        print("Error fetching data:", response.status_code)
        return []

    return response.json()


def parse_finnhub_articles(news):
    parsed = []

    for article in news:
        try:
            parsed.append(
                {
                    "title": clean_text(article.get("headline")),
                    "url": article.get("url"),
                    "source": article.get("source"),
                    "published": datetime.fromtimestamp(article.get("datetime")),
                    "summary": clean_text(article.get("summary")),
                }
            )
        except Exception:
            continue

    return parsed


def deduplicate_articles(articles):
    seen = set()
    unique = []

    for article in articles:
        title = article["title"]
        if title and title not in seen:
            seen.add(title)
            unique.append(article)

    return unique


def scrape_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return clean_text(article.text)
    except Exception:
        return None


def get_finbert_sentiment(text):
    if not text:
        return None, None, None

    result = finbert(text[:512])[0]

    label = result["label"]
    confidence = result["score"]
    score = SENTIMENT_TO_SCORE.get(label.lower(), 0)

    return label, confidence, score


def build_article_dataset(ticker, api_key, days_back=7, delay=1, max_articles=10):
    print(f"\nFetching news for {ticker}...")

    raw_news = get_stock_news(ticker, api_key, days_back)
    print(f"Raw articles pulled: {len(raw_news)}")

    parsed_articles = parse_finnhub_articles(raw_news)
    unique_articles = deduplicate_articles(parsed_articles)[:max_articles]
    print(f"Processing {len(unique_articles)} unique articles...")

    data = []

    for index, article in enumerate(unique_articles, start=1):
        print(f"[{index}/{len(unique_articles)}] Scraping: {article['title']}")

        text = scrape_text(article["url"])
        if text is None or len(text) < 200:
            continue

        sentiment_input = text if len(text) >= 512 else article["summary"]
        label, confidence, score = get_finbert_sentiment(sentiment_input)

        data.append(
            {
                "ticker": ticker,
                "title": article["title"],
                "source": article["source"],
                "published": article["published"],
                "published_date": article["published"].date(),
                "url": article["url"],
                "summary": article["summary"],
                "sentiment_label": label,
                "sentiment_confidence": confidence,
                "sentiment_score": score,
                "text": text,
                "text_length": len(text),
            }
        )

        time.sleep(delay)

    return pd.DataFrame(data)


def score_to_outlook_label(score):
    if score >= 0.6:
        return "Bullish"
    if score <= 0.4:
        return "Bearish"
    return "Neutral"


def summarize_sentiment(article_df):
    if article_df.empty:
        return {
            "article_count": 0,
            "average_sentiment_score": 0.0,
            "weighted_sentiment_score": 0.0,
            "overall_sentiment_score": 0.0,
            "overall_sentiment_label": "Neutral",
            "positive_articles": 0,
            "negative_articles": 0,
            "neutral_articles": 0,
        }

    working_df = article_df.copy()
    most_recent = pd.to_datetime(working_df["published"]).max()
    article_age_days = (
        (most_recent - pd.to_datetime(working_df["published"])).dt.total_seconds() / 86400
    ).clip(lower=0)

    # More recent, higher-confidence articles should contribute more to the aggregate view.
    recency_weight = 1 / (1 + article_age_days)
    confidence_weight = working_df["sentiment_confidence"].fillna(0)
    combined_weight = recency_weight * confidence_weight

    if combined_weight.sum() == 0:
        weighted_average = float(working_df["sentiment_score"].mean())
    else:
        weighted_average = float(
            (working_df["sentiment_score"] * combined_weight).sum() / combined_weight.sum()
        )

    counts = article_df["sentiment_label"].str.lower().value_counts()
    overall_score = round(weighted_average, 4)

    return {
        "article_count": int(len(article_df)),
        "average_sentiment_score": round(float(article_df["sentiment_score"].mean()), 4),
        "weighted_sentiment_score": round(weighted_average, 4),
        "overall_sentiment_score": overall_score,
        "overall_sentiment_label": score_to_outlook_label(overall_score),
        "positive_articles": int(counts.get("positive", 0)),
        "negative_articles": int(counts.get("negative", 0)),
        "neutral_articles": int(counts.get("neutral", 0)),
    }


def merge_analysis_frames(*frames):
    valid_frames = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    combined = pd.concat(valid_frames, ignore_index=True, sort=False)
    if "published" in combined.columns:
        combined["published"] = pd.to_datetime(combined["published"], errors="coerce")
    if "published_date" not in combined.columns and "published" in combined.columns:
        combined["published_date"] = pd.to_datetime(combined["published"], errors="coerce").dt.date

    return combined


def extract_key_phrases(article_df, max_phrases=6):
    if article_df.empty:
        return []

    stop_words = {
        "the", "and", "for", "with", "that", "from", "this", "after", "have", "has",
        "into", "their", "about", "over", "will", "amid", "more", "than", "said",
        "says", "stock", "stocks", "company", "shares", "could", "would", "were",
        "been", "because", "while", "where", "when", "which", "whose", "also",
        "into", "through", "around", "under", "between", "during", "they", "them",
        "news", "article", "market", "markets",
    }

    text_blob = " ".join(
        article_df["title"].fillna("").tolist() + article_df["summary"].fillna("").tolist()
    ).lower()
    tokens = re.findall(r"\b[a-z]{4,}\b", text_blob)

    filtered_tokens = [token for token in tokens if token not in stop_words]
    return [token for token, _ in Counter(filtered_tokens).most_common(max_phrases)]


def build_overall_summary(article_df, sentiment_summary):
    if article_df.empty:
        return "No usable articles were scraped, so there is not enough news context to form an outlook."

    sorted_df = article_df.sort_values("published", ascending=False).copy()
    recent_titles = sorted_df["title"].head(3).tolist()
    top_sources = sorted_df["source"].fillna("Unknown").value_counts().head(3).index.tolist()
    key_phrases = extract_key_phrases(sorted_df)

    tone = sentiment_summary["overall_sentiment_label"].lower()
    score = sentiment_summary["overall_sentiment_score"]
    article_count = sentiment_summary["article_count"]

    source_text = ", ".join(top_sources) if top_sources else "mixed financial sources"
    topic_text = ", ".join(key_phrases[:4]) if key_phrases else "recent company developments"
    title_text = " | ".join(recent_titles[:2]) if recent_titles else "No major headline captured"

    return (
        f"Across {article_count} recent articles, the overall news tone for the stock looks {tone} "
        f"with an aggregate sentiment score of {score} on a 0 to 1 scale, where 0.5 is neutral. "
        f"The main narrative themes appearing across "
        f"coverage are {topic_text}. Coverage has been driven largely by {source_text}. Recent headlines "
        f"suggest the current story is centered on: {title_text}."
    )


def build_source_summary(dataset_name, analysis_df, sentiment_summary):
    if analysis_df.empty:
        return f"No usable {dataset_name.lower()} data was available for this ticker."

    sorted_df = analysis_df.sort_values("published", ascending=False).copy()
    recent_titles = sorted_df["title"].fillna("").head(3).tolist()
    top_sources = sorted_df["source"].fillna("Unknown").value_counts().head(3).index.tolist()
    key_phrases = extract_key_phrases(sorted_df)

    tone = sentiment_summary["overall_sentiment_label"].lower()
    score = sentiment_summary["overall_sentiment_score"]
    item_count = sentiment_summary["article_count"]

    source_text = ", ".join(top_sources) if top_sources else dataset_name.lower()
    topic_text = ", ".join(key_phrases[:4]) if key_phrases else "recent developments"
    title_text = " | ".join([title for title in recent_titles[:2] if title]) or "No major item captured"

    return (
        f"Across {item_count} {dataset_name.lower()} items, the tone looks {tone} with an aggregate "
        f"sentiment score of {score} on a 0 to 1 scale, where 0.5 is neutral. "
        f"The dominant themes are {topic_text}. Coverage is primarily sourced from {source_text}. "
        f"Recent highlights include: {title_text}."
    )


def build_llm_article_context(article_df, max_articles=6, max_chars_per_article=1200):
    if article_df.empty:
        return "No article content was available."

    def select_balanced_items(frame):
        if "analysis_type" not in frame.columns:
            return frame.sort_values("published", ascending=False).head(max_articles).copy()

        working = frame.copy()
        working["analysis_type"] = working["analysis_type"].fillna("other")
        available_types = [
            analysis_type
            for analysis_type in ["news", "edgar", "social", "other"]
            if analysis_type in working["analysis_type"].unique()
        ]

        if len(available_types) <= 1:
            return working.sort_values("published", ascending=False).head(max_articles).copy()

        per_type_limit = max(1, max_articles // len(available_types))
        selected_frames = []

        for analysis_type in available_types:
            type_slice = (
                working.loc[working["analysis_type"] == analysis_type]
                .sort_values("published", ascending=False)
                .head(per_type_limit)
                .copy()
            )
            if not type_slice.empty:
                selected_frames.append(type_slice)

        balanced_df = pd.concat(selected_frames, ignore_index=True, sort=False)

        if len(balanced_df) < max_articles:
            chosen_titles = set(balanced_df["title"].fillna("").tolist())
            remaining = (
                working.loc[~working["title"].fillna("").isin(chosen_titles)]
                .sort_values("published", ascending=False)
                .head(max_articles - len(balanced_df))
                .copy()
            )
            if not remaining.empty:
                balanced_df = pd.concat([balanced_df, remaining], ignore_index=True, sort=False)

        return balanced_df.head(max_articles).copy()

    sections = []
    trimmed_df = select_balanced_items(article_df)

    grouped_context = (
        "analysis_type" in trimmed_df.columns
        and trimmed_df["analysis_type"].fillna("").nunique() > 1
    )

    if grouped_context:
        display_names = {
            "news": "Stock News",
            "edgar": "EDGAR Filings",
            "social": "Social Posts",
            "other": "Other Sources",
        }
        section_index = 1
        for analysis_type in ["news", "edgar", "social", "other"]:
            type_df = trimmed_df.loc[
                trimmed_df["analysis_type"].fillna("other") == analysis_type
            ].copy()
            if type_df.empty:
                continue

            sections.append(f"{display_names.get(analysis_type, analysis_type.title())}")
            for article in type_df.itertuples(index=False):
                text = safe_string(getattr(article, "text", ""))
                trimmed_text = text[:max_chars_per_article]
                sections.append(
                    (
                        f"Item {section_index}\n"
                        f"Title: {safe_string(getattr(article, 'title', ''))}\n"
                        f"Source: {safe_string(getattr(article, 'source', ''))}\n"
                        f"Published: {safe_string(getattr(article, 'published', ''))}\n"
                        f"FinBERT label: {safe_string(getattr(article, 'sentiment_label', ''))}\n"
                        f"FinBERT score: {safe_string(getattr(article, 'sentiment_score', ''))}\n"
                        f"Summary: {safe_string(getattr(article, 'summary', ''))}\n"
                        f"Body excerpt: {trimmed_text}"
                    )
                )
                section_index += 1

        return "\n\n".join(sections)

    for index, article in enumerate(trimmed_df.itertuples(index=False), start=1):
        text = safe_string(getattr(article, "text", ""))
        trimmed_text = text[:max_chars_per_article]
        sections.append(
            (
                f"Article {index}\n"
                f"Title: {safe_string(getattr(article, 'title', ''))}\n"
                f"Source: {safe_string(getattr(article, 'source', ''))}\n"
                f"Published: {safe_string(getattr(article, 'published', ''))}\n"
                f"FinBERT label: {safe_string(getattr(article, 'sentiment_label', ''))}\n"
                f"FinBERT score: {safe_string(getattr(article, 'sentiment_score', ''))}\n"
                f"Summary: {safe_string(getattr(article, 'summary', ''))}\n"
                f"Body excerpt: {trimmed_text}"
            )
        )

    return "\n\n".join(sections)


def get_local_summarizer():
    global summarizer
    global summarizer_tokenizer

    if summarizer is None:
        model_name = os.getenv("LOCAL_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
        summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"Local summarizer model loaded: {model_name}")

    return summarizer, summarizer_tokenizer


def summarize_text_locally(text, max_length=180, min_length=60):
    if not text:
        return ""

    model, tokenizer = get_local_summarizer()
    prepared_text = "summarize: " + text[:3500]

    inputs = tokenizer(
        prepared_text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    output_tokens = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()


def generate_local_llm_insights(ticker, article_df, sentiment_summary):
    if article_df.empty:
        return None

    article_context = build_llm_article_context(article_df)
    llm_view_df = article_df.sort_values("published", ascending=False).copy()
    if "analysis_type" in llm_view_df.columns and llm_view_df["analysis_type"].fillna("").nunique() > 1:
        selected_titles = []
        for analysis_type in ["news", "edgar", "social", "other"]:
            type_titles = (
                llm_view_df.loc[llm_view_df["analysis_type"].fillna("other") == analysis_type, "title"]
                .fillna("")
                .head(2)
                .tolist()
            )
            selected_titles.extend([safe_string(title) for title in type_titles if safe_string(title)])
        selected_titles = selected_titles[:6]
        selected_summaries = []
        for analysis_type in ["news", "edgar", "social", "other"]:
            type_summaries = (
                llm_view_df.loc[llm_view_df["analysis_type"].fillna("other") == analysis_type, "summary"]
                .fillna("")
                .head(2)
                .tolist()
            )
            selected_summaries.extend(
                [safe_string(summary) for summary in type_summaries if safe_string(summary)]
            )
        selected_summaries = selected_summaries[:6]
    else:
        selected_titles = [safe_string(title) for title in llm_view_df["title"].fillna("").head(5).tolist()]
        selected_summaries = [
            safe_string(summary) for summary in llm_view_df["summary"].fillna("").head(5).tolist()
        ]

    title_text = " ".join(selected_titles)
    summary_text = " ".join(selected_summaries)
    key_phrases = extract_key_phrases(article_df)
    score = sentiment_summary["overall_sentiment_score"]
    label = sentiment_summary["overall_sentiment_label"]

    main_points_summary = summarize_text_locally(
        article_context,
        max_length=140,
        min_length=50,
    )

    outlook_prompt_text = (
        f"{ticker} stock outlook. "
        f"Overall sentiment is {label} with score {score} on a 0 to 1 scale. "
        f"Key themes include: {', '.join(key_phrases[:5])}. "
        f"Headline context: {title_text}. "
        f"Article summaries: {summary_text}."
    )
    overall_outlook = summarize_text_locally(
        outlook_prompt_text,
        max_length=120,
        min_length=40,
    )

    main_points = []
    for title in selected_titles[:4]:
        if title:
            main_points.append(title.strip())

    bullish_points = llm_view_df.loc[
        llm_view_df["sentiment_label"].str.lower() == "positive", "title"
    ].head(2).fillna("").tolist()
    bearish_points = llm_view_df.loc[
        llm_view_df["sentiment_label"].str.lower() == "negative", "title"
    ].head(2).fillna("").tolist()

    return {
        "overall_outlook": overall_outlook,
        "main_points_summary": main_points_summary,
        "main_points": main_points,
        "bullish_points": [point for point in bullish_points if point],
        "bearish_points": [point for point in bearish_points if point],
    }


def normalize_edgar_dataset(edgar_df):
    if edgar_df is None or edgar_df.empty:
        return pd.DataFrame()

    normalized = edgar_df.copy()
    normalized["published"] = pd.to_datetime(normalized["published"], errors="coerce")
    normalized["published_date"] = pd.to_datetime(normalized["published"], errors="coerce").dt.date
    normalized["sentiment_score"] = normalized["sentiment_label"].fillna("").str.lower().map(
        SENTIMENT_TO_SCORE
    )
    normalized["analysis_type"] = "edgar"
    return normalized


def get_edgar_analysis(ticker, days_back=30, max_filings=5):
    try:
        edgar_df = build_edgar_dataset(
            ticker=ticker,
            days_back=days_back,
            max_filings=max_filings,
        )
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    return normalize_edgar_dataset(edgar_df), None


def annotate_news_dataset(article_df):
    if article_df is None or article_df.empty:
        return pd.DataFrame()

    annotated = article_df.copy()
    annotated["analysis_type"] = "news"
    return annotated


def get_ticker_sector(ticker):
    stock = yf.Ticker(ticker)

    info = {}
    try:
        info = stock.info
    except Exception:
        info = {}

    sector = info.get("sector") or info.get("category")
    if not sector:
        raise ValueError(f"Could not determine sector for ticker {ticker}.")

    return sector


def analyze_ticker_news(ticker, api_key, days_back=7, max_articles=10):
    try:
        sector = get_ticker_sector(ticker)
    except Exception as exc:
        sector = f"Unknown ({exc})"

    article_df = build_article_dataset(
        ticker=ticker,
        api_key=api_key,
        days_back=days_back,
        max_articles=max_articles,
    )
    article_df = annotate_news_dataset(article_df)

    edgar_df, edgar_error = get_edgar_analysis(
        ticker=ticker,
        days_back=max(days_back, 30),
        max_filings=max_articles,
    )

    combined_df = merge_analysis_frames(article_df, edgar_df)

    news_sentiment_summary = summarize_sentiment(article_df)
    edgar_sentiment_summary = summarize_sentiment(edgar_df)
    sentiment_summary = summarize_sentiment(combined_df)

    news_overall_summary = build_overall_summary(article_df, news_sentiment_summary)
    edgar_overall_summary = build_source_summary("EDGAR filing", edgar_df, edgar_sentiment_summary)
    overall_summary = build_source_summary("combined news and filing", combined_df, sentiment_summary)

    llm_insights = None
    article_llm_insights = None
    edgar_llm_insights = None
    llm_error = None
    article_llm_error = None
    edgar_llm_error = None

    try:
        llm_insights = generate_local_llm_insights(ticker, combined_df, sentiment_summary)
    except Exception as exc:
        llm_error = str(exc)

    try:
        article_llm_insights = generate_local_llm_insights(
            ticker,
            article_df,
            news_sentiment_summary,
        )
    except Exception as exc:
        article_llm_error = str(exc)

    try:
        edgar_llm_insights = generate_local_llm_insights(
            ticker,
            edgar_df,
            edgar_sentiment_summary,
        )
    except Exception as exc:
        edgar_llm_error = str(exc)

    return {
        "ticker": ticker,
        "sector": sector,
        "article_df": article_df,
        "edgar_df": edgar_df,
        "combined_df": combined_df,
        "sentiment_summary": sentiment_summary,
        "news_sentiment_summary": news_sentiment_summary,
        "edgar_sentiment_summary": edgar_sentiment_summary,
        "overall_summary": overall_summary,
        "news_overall_summary": news_overall_summary,
        "edgar_overall_summary": edgar_overall_summary,
        "llm_insights": llm_insights,
        "article_llm_insights": article_llm_insights,
        "edgar_llm_insights": edgar_llm_insights,
        "llm_error": llm_error,
        "article_llm_error": article_llm_error,
        "edgar_llm_error": edgar_llm_error,
        "edgar_error": edgar_error,
    }


def main():
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("Set the FINNHUB_API_KEY environment variable before running this script.")

    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()
    result = analyze_ticker_news(ticker, api_key, days_back=7, max_articles=10)
    article_df = result["article_df"]
    sentiment_summary = result["sentiment_summary"]
    overall_summary = result["overall_summary"]
    llm_insights = result["llm_insights"]
    llm_error = result["llm_error"]

    print(f"\nTicker: {ticker}")
    print(f"Sector: {result['sector']}")
    print("Sentiment summary:")
    print(sentiment_summary)
    print("\nOverall outlook:")
    print(overall_summary)

    if llm_insights:
        print("\nLocal model stock outlook:")
        print(llm_insights["overall_outlook"])
        print("\nLocal model main points summary:")
        print(llm_insights["main_points_summary"])

    if llm_error:
        print("\nLocal summary output unavailable:")
        print(llm_error)

    print("\nArticle sample:")
    print(article_df.head())

    output_file = f"{ticker.lower()}_articles.csv"
    article_df.to_csv(output_file, index=False)
    print(f"\nSaved article dataset to {output_file}")


if __name__ == "__main__":
    main()
