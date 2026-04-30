import yfinance as yf
from datetime import datetime, timedelta
from newspaper import Article
import pandas as pd
import time


def get_yfinance_articles(ticker):
    """
    Pull raw news articles from Yahoo Finance
    """
    tk = yf.Ticker(ticker)
    return tk.news


def filter_articles_by_time(articles, days_back=7):
    """
    Keep only articles within the specified time window
    """
    cutoff = datetime.now() - timedelta(days=days_back)

    filtered = []
    for a in articles:
        print(a)
        pub_time = datetime.fromtimestamp(a['providerPublishTime'])
        if pub_time >= cutoff:
            filtered.append(a)

    return filtered


def scrape_article_text(url):
    """
    Extract full article text from URL
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None


def build_dataset(ticker, days_back=7, delay=1):
    """
    Full pipeline:
    - Fetch articles
    - Filter by time
    - Scrape content
    - Return structured DataFrame
    """
    print(f"Fetching articles for {ticker}...")

    raw_articles = get_yfinance_articles(ticker)
    filtered_articles = filter_articles_by_time(raw_articles, days_back)

    print(f"Found {len(filtered_articles)} articles in last {days_back} days")

    data = []

    for i, a in enumerate(filtered_articles):
        url = a.get("link")
        title = a.get("title")
        source = a.get("publisher")
        pub_time = datetime.fromtimestamp(a['providerPublishTime'])

        print(f"[{i+1}/{len(filtered_articles)}] Scraping: {title}")

        text = scrape_article_text(url)

        # Skip if scraping failed or too little content
        if text is None or len(text) < 200:
            continue

        data.append({
            "ticker": ticker,
            "title": title,
            "source": source,
            "published": pub_time,
            "url": url,
            "text": text
        })

        time.sleep(delay)  # avoid rate limiting

    df = pd.DataFrame(data)

    print(f"\nFinal dataset size: {len(df)} articles")

    return df


# ===== RUN =====
if __name__ == "__main__":
    df = build_dataset("AAPL", days_back=7)
    print(df.head())