import html
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import pandas as pd

from WebScrape import (
    analyze_ticker_news,
    build_source_summary,
    generate_local_llm_insights,
    merge_analysis_frames,
    summarize_sentiment,
)
from live_stock_prediction import predict_with_live_stock_model
from stockScrape import analyze_social_sentiment


HOST = "127.0.0.1"
PORT = 8000


def safe_text(value):
    return html.escape("" if value is None else str(value))


def sentiment_badge_class(label):
    normalized = (label or "").strip().lower()
    if normalized == "positive":
        return "pill-positive"
    if normalized == "negative":
        return "pill-negative"
    if normalized == "neutral":
        return "pill-neutral"
    return "pill-default"


def normalize_social_for_combined_snapshot(social_df):
    if social_df is None or getattr(social_df, "empty", True):
        return pd.DataFrame()

    normalized = social_df.copy()
    normalized["sentiment_score"] = normalized["sentiment_label"].fillna("").str.lower().map(
        {"positive": 1.0, "neutral": 0.5, "negative": 0.0}
    ).fillna(0.5)
    if "confidence" in normalized.columns:
        normalized["sentiment_confidence"] = pd.to_numeric(
            normalized["confidence"],
            errors="coerce",
        ).fillna(1.0)
    else:
        normalized["sentiment_confidence"] = 1.0
    normalized["analysis_type"] = "social"
    return normalized


def render_content_cards(content_df, empty_message, link_label):
    if content_df is None or getattr(content_df, "empty", True):
        return f"<p class='muted'>{safe_text(empty_message)}</p>"

    cards = []
    for item in content_df.head(6).itertuples(index=False):
        sentiment_label = getattr(item, "sentiment_label", "Unknown")
        sentiment_class = sentiment_badge_class(sentiment_label)

        cards.append(
            f"""
            <article class="article-card">
                <h3>{safe_text(getattr(item, "title", ""))}</h3>
                <div class="article-footer">
                    <span class="pill {sentiment_class}">{safe_text(sentiment_label)}</span>
                    <a href="{safe_text(getattr(item, "url", "#"))}" target="_blank" rel="noreferrer">{safe_text(link_label)}</a>
                </div>
            </article>
            """
        )
    return "".join(cards)


def render_sentiment_panel(title, sentiment, summary_text):
    return f"""
        <div class="panel">
            <h2>{safe_text(title)}</h2>
            <p>{safe_text(summary_text)}</p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="metric-label">Overall Score</div>
                    <div class="metric-value">{safe_text(sentiment['overall_sentiment_score'])}</div>
                </div>
                <div class="stat-card">
                    <div class="metric-label">Label</div>
                    <div class="metric-value">{safe_text(sentiment['overall_sentiment_label'])}</div>
                </div>
                <div class="stat-card">
                    <div class="metric-label">Items Analyzed</div>
                    <div class="metric-value">{safe_text(sentiment['article_count'])}</div>
                </div>
                <div class="stat-card">
                    <div class="metric-label">Positive Items</div>
                    <div class="metric-value">{safe_text(sentiment['positive_articles'])}</div>
                </div>
            </div>
        </div>
    """


def render_ai_panel(title, insights, error_message):
    if insights:
        return f"""
            <div class="panel">
                <h2>{safe_text(title)}</h2>
                <p>{safe_text(insights['overall_outlook'])}</p>
                <h3>Main Points Summary</h3>
                <p>{safe_text(insights['main_points_summary'])}</p>
            </div>
        """

    if error_message:
        return f"""
            <div class="panel">
                <h2>{safe_text(title)}</h2>
                <p class="muted">AI insights unavailable: {safe_text(error_message)}</p>
            </div>
        """

    return f"""
        <div class="panel">
            <h2>{safe_text(title)}</h2>
            <p class="muted">No content was available for AI-generated insights.</p>
        </div>
    """


def render_results(ticker, result):
    social_df = result.get("social_df")
    combined_snapshot_df = merge_analysis_frames(
        result["article_df"],
        result["edgar_df"],
        normalize_social_for_combined_snapshot(social_df),
    )
    sentiment = summarize_sentiment(combined_snapshot_df)
    news_sentiment = result["news_sentiment_summary"]
    edgar_sentiment = result["edgar_sentiment_summary"]
    social_sentiment = result.get("social_sentiment_summary", {})
    combined_summary = build_source_summary(
        "combined news, filing, and social",
        combined_snapshot_df,
        sentiment,
    )
    llm_insights = result["llm_insights"]
    prediction = result.get("prediction_result")
    prediction_error = result.get("prediction_error")

    confidence_text = "Unavailable"
    if prediction and prediction.get("probability_up") is not None:
        confidence_text = f"{prediction['probability_up']:.3f}"

    prediction_message_html = "<p class='muted'>Prediction unavailable for this ticker's selected stock model.</p>"
    if prediction_error:
        prediction_message_html = f"<p class='muted'>{safe_text(prediction_error)}</p>"

    prediction_html = f"""
        <div class="panel">
            <h2>Model Prediction</h2>
            {prediction_message_html}
        </div>
    """
    if prediction:
        prediction_html = f"""
            <div class="panel hero-panel">
                <h2>Next Day Movement Prediction</h2>
                <div class="hero-grid">
                    <div>
                        <div class="metric-label">Ticker</div>
                        <div class="metric-value">{safe_text(ticker)}</div>
                    </div>
                    <div>
                        <div class="metric-label">Expected Direction</div>
                        <div class="metric-value">{safe_text(prediction['prediction_label'])}</div>
                    </div>
                    <div>
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value">{safe_text(confidence_text)}</div>
                    </div>
                </div>
            </div>
        """

    return f"""
        {prediction_html}
        <div class="content-grid">
            {render_sentiment_panel("Combined Sentiment Snapshot", sentiment, combined_summary)}
            {render_ai_panel("Combined AI Insights", llm_insights, result.get("llm_error"))}
            {render_sentiment_panel("News Sentiment", news_sentiment, result["news_overall_summary"])}
            {render_ai_panel("News AI Insights", result.get("article_llm_insights"), result.get("article_llm_error"))}
            {render_sentiment_panel("EDGAR Filing Sentiment", edgar_sentiment, result["edgar_overall_summary"])}
            {render_ai_panel("EDGAR Filing AI Insights", result.get("edgar_llm_insights"), result.get("edgar_llm_error"))}
            {render_sentiment_panel("Social Media Sentiment", social_sentiment, result.get("social_overall_summary", "No usable social posts were analyzed for this ticker."))}
            <div class="panel panel-wide">
                <h2>Recent Articles</h2>
                <div class="articles-grid">
                    {render_content_cards(result['article_df'], "No usable articles were scraped for this ticker.", "Open article")}
                </div>
            </div>
            <div class="panel panel-wide">
                <h2>Recent Social Posts</h2>
                <div class="articles-grid">
                    {render_content_cards(result.get('social_df'), "No usable Reddit posts were found for this ticker.", "Open post")}
                </div>
            </div>
            <div class="panel panel-wide">
                <h2>Recent SEC Filings</h2>
                <div class="articles-grid">
                    {render_content_cards(result['edgar_df'], "No recent 8-K, 10-Q, or 10-K filings were captured for this ticker.", "Open filing")}
                </div>
            </div>
        </div>
    """


def render_page(results_html="", error_message="", ticker_value="", article_count_value="10"):
    error_html = ""
    if error_message:
        error_html = f"<div class='error-banner'>{safe_text(error_message)}</div>"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Stock Outlook Demo</title>
        <style>
            :root {{
                --bg: #f4efe6;
                --panel: #fffaf2;
                --ink: #1f2937;
                --muted: #5b6472;
                --accent: #0f766e;
                --accent-2: #c2410c;
                --line: #e7dbc8;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: Georgia, "Times New Roman", serif;
                color: var(--ink);
                background:
                    radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 32%),
                    radial-gradient(circle at top right, rgba(194,65,12,0.10), transparent 28%),
                    var(--bg);
            }}
            .wrap {{
                max-width: 1180px;
                margin: 0 auto;
                padding: 32px 20px 48px;
            }}
            .masthead {{
                background: linear-gradient(135deg, rgba(15,118,110,0.94), rgba(17,24,39,0.96));
                color: white;
                border-radius: 24px;
                padding: 28px;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
            }}
            .masthead h1 {{
                margin: 0 0 10px;
                font-size: clamp(2rem, 5vw, 3.6rem);
                line-height: 1;
            }}
            .masthead p {{
                margin: 0;
                max-width: 760px;
                color: rgba(255,255,255,0.82);
                font-size: 1.05rem;
            }}
            .search-panel {{
                margin-top: 22px;
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 18px;
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                align-items: center;
            }}
            .search-panel input {{
                flex: 1 1 240px;
                min-width: 0;
                padding: 14px 16px;
                border-radius: 14px;
                border: 1px solid #cbd5e1;
                font-size: 1rem;
                background: white;
            }}
            .search-panel button {{
                padding: 14px 20px;
                border: none;
                border-radius: 14px;
                background: var(--accent-2);
                color: white;
                font-weight: 700;
                cursor: pointer;
            }}
            .hero-panel, .panel {{
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 22px;
                margin-top: 22px;
                box-shadow: 0 10px 30px rgba(148, 163, 184, 0.08);
            }}
            .hero-grid, .stats-grid, .articles-grid, .content-grid {{
                display: grid;
                gap: 16px;
            }}
            .hero-grid {{
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            }}
            .stats-grid {{
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            }}
            .content-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .panel-wide {{
                grid-column: 1 / -1;
            }}
            .articles-grid {{
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            }}
            .article-card, .stat-card {{
                background: white;
                border: 1px solid #efe5d8;
                border-radius: 16px;
                padding: 16px;
            }}
            .article-footer {{
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                justify-content: space-between;
                color: var(--muted);
                font-size: 0.9rem;
            }}
            .article-card h3 {{
                margin: 0 0 12px;
                font-size: 1.05rem;
                line-height: 1.3;
            }}
            .article-card p, .panel p {{
                line-height: 1.55;
            }}
            .metric-label {{
                color: var(--muted);
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }}
            .metric-value {{
                margin-top: 6px;
                font-size: 1.5rem;
                font-weight: 700;
            }}
            .pill {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: rgba(15,118,110,0.10);
                color: var(--accent);
                font-size: 0.85rem;
                text-transform: capitalize;
            }}
            .pill-positive {{
                background: #dcfce7;
                color: #166534;
            }}
            .pill-negative {{
                background: #fee2e2;
                color: #991b1b;
            }}
            .pill-neutral {{
                background: #e5e7eb;
                color: #374151;
            }}
            .pill-default {{
                background: #f3f4f6;
                color: #4b5563;
            }}
            .muted {{
                color: var(--muted);
            }}
            .error-banner {{
                margin-top: 18px;
                padding: 14px 16px;
                border-radius: 14px;
                background: #fff1f2;
                color: #9f1239;
                border: 1px solid #fecdd3;
            }}
            a {{
                color: var(--accent-2);
            }}
            @media (max-width: 760px) {{
                .content-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <section class="masthead">
                <h1>Trader Lens</h1>
                <p>Enter a stock ticker and choose how many items to pull per source for company news, SPY news, SEC EDGAR filings, and social posts.</p>
                <form class="search-panel" method="post">
                    <input
                        type="text"
                        name="ticker"
                        placeholder="Try AAPL or MSFT"
                        value="{safe_text(ticker_value)}"
                        required
                    >
                    <input
                        type="number"
                        name="article_count"
                        min="1"
                        max="25"
                        placeholder="Items per source"
                        value="{safe_text(article_count_value)}"
                        required
                    >
                    <button type="submit">Analyze Ticker</button>
                </form>
                {error_html}
            </section>
            {results_html}
        </div>
    </body>
    </html>
    """


class StockAppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond(render_page())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        params = parse_qs(body)
        ticker = params.get("ticker", [""])[0].strip().upper()
        article_count_text = params.get("article_count", ["10"])[0].strip()

        if not ticker:
            self.respond(render_page(error_message="Please enter a ticker symbol."))
            return

        try:
            article_count = int(article_count_text)
        except ValueError:
            self.respond(
                render_page(
                    error_message="Please enter a valid number of items per source.",
                    ticker_value=ticker,
                    article_count_value=article_count_text,
                )
            )
            return

        if article_count < 1 or article_count > 25:
            self.respond(
                render_page(
                    error_message="Items per source must be between 1 and 25.",
                    ticker_value=ticker,
                    article_count_value=article_count_text,
                )
            )
            return

        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            self.respond(
                render_page(
                    error_message="FINNHUB_API_KEY is not set in the environment.",
                    ticker_value=ticker,
                    article_count_value=article_count_text,
                )
            )
            return

        try:
            result = analyze_ticker_news(
                ticker,
                api_key,
                days_back=7,
                max_articles=article_count,
            )
            social_error = None
            try:
                social_result = analyze_social_sentiment(
                    ticker,
                    post_limit=article_count,
                    sort="hot",
                    time_filter="week",
                )
            except Exception as exc:
                social_result = {
                    "social_df": None,
                    "social_sentiment_summary": {
                        "article_count": 0,
                        "average_sentiment_score": 0.0,
                        "weighted_sentiment_score": 0.0,
                        "overall_sentiment_score": 0.0,
                        "overall_sentiment_label": "Neutral",
                        "positive_articles": 0,
                        "negative_articles": 0,
                        "neutral_articles": 0,
                    },
                    "social_overall_summary": "Social sentiment is currently unavailable for this ticker.",
                    "social_metadata": {},
                }
                social_error = str(exc)

            result.update(social_result)

            combined_snapshot_df = merge_analysis_frames(
                result["article_df"],
                result["edgar_df"],
                normalize_social_for_combined_snapshot(result.get("social_df")),
            )
            combined_snapshot_summary = summarize_sentiment(combined_snapshot_df)
            result["sentiment_summary"] = combined_snapshot_summary
            result["overall_summary"] = build_source_summary(
                "combined news, filing, and social",
                combined_snapshot_df,
                combined_snapshot_summary,
            )
            try:
                result["llm_insights"] = generate_local_llm_insights(
                    ticker,
                    combined_snapshot_df,
                    combined_snapshot_summary,
                )
                result["llm_error"] = None
            except Exception as exc:
                result["llm_insights"] = None
                result["llm_error"] = str(exc)

            prediction_result = None
            prediction_error = None
            try:
                prediction_result = predict_with_live_stock_model(
                    ticker,
                    api_key=api_key,
                    max_news_articles=article_count,
                )
            except Exception as exc:
                prediction_error = str(exc)

            result["prediction_result"] = prediction_result
            result["prediction_error"] = prediction_error

            results_html = render_results(ticker, result)
            if result.get("edgar_error"):
                results_html = (
                    f"<div class='error-banner'>{safe_text('EDGAR filings unavailable: ' + result['edgar_error'])}</div>"
                    + results_html
                )
            if prediction_error:
                results_html = (
                    f"<div class='error-banner'>{safe_text(prediction_error)}</div>" + results_html
                )
            if social_error:
                results_html = (
                    f"<div class='error-banner'>{safe_text('Social sentiment unavailable: ' + social_error)}</div>"
                    + results_html
                )

            self.respond(
                render_page(
                    results_html=results_html,
                    ticker_value=ticker,
                    article_count_value=str(article_count),
                )
            )
        except Exception as exc:
            self.respond(
                render_page(
                    error_message=str(exc),
                    ticker_value=ticker,
                    article_count_value=article_count_text,
                )
            )

    def log_message(self, format, *args):
        return

    def respond(self, html_body):
        encoded = html_body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def main():
    server = HTTPServer((HOST, PORT), StockAppHandler)
    print(f"Serving Stock Outlook app at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
