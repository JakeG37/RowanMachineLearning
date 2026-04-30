# ──────────────────────────────────────────────────────────────────────────────
# Standard Library
# ──────────────────────────────────────────────────────────────────────────────
import re
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

# ──────────────────────────────────────────────────────────────────────────────
# Third-Party
# ──────────────────────────────────────────────────────────────────────────────
import requests
import torch
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, Pipeline


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  —  No credentials needed. Only edit these values.
# ══════════════════════════════════════════════════════════════════════════════

# Reddit requires a descriptive User-Agent string — do not leave this generic
USER_AGENT: str = "stock_sentiment_bot/1.0 (research project)"

# Subreddits to search — ordered by signal quality for stock sentiment
TARGET_SUBREDDITS: list[str] = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "StockMarket",
]

# FinBERT on Hugging Face — swap for a local fine-tuned checkpoint path if needed
FINBERT_MODEL: str = "ProsusAI/finbert"

# Device: auto-detects CUDA GPU, falls back to CPU if unavailable
DEVICE: int = 0 if torch.cuda.is_available() else -1

# Sentiment label to numeric score mapping
SENTIMENT_SCORE_MAP: dict[str, int] = {
    "positive":  1.0,
    "neutral":   0.5,
    "negative":  0.0,
}

# Minimum character length for a post to pass quality filtering
MIN_TEXT_LENGTH: int = 20

# FinBERT inference batch size — reduce to 8 if running out of GPU VRAM
BATCH_SIZE: int = 16

# Seconds to wait between subreddit requests — respects Reddit rate limit
REQUEST_DELAY: float = 1.2

# Reddit public JSON base URL
REDDIT_BASE_URL: str = "https://www.reddit.com"


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
_FINBERT_PIPE: Pipeline | None = None


# ══════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RedditPost:
    """
    Represents a single Reddit post normalised and ready for NLP inference.
    Each field maps directly to a column in the output CSV.

    Fields
    ------
    post_id       : Unique Reddit ID (base-36 string)
    ticker        : Stock ticker symbol, upper-cased
    subreddit     : Community the post was fetched from
    title         : Raw post title
    body          : Raw post body (selftext); empty string if link-only post
    combined_text : Cleaned title + body, truncated to 512 chars for FinBERT
    author        : Reddit username, or "[deleted]" if account removed
    score         : Net upvote count at time of fetch
    upvote_ratio  : Fraction of votes that are upvotes (0.0 to 1.0)
    num_comments  : Comment count — proxy for engagement and controversy
    url           : Full permalink to the post
    created_utc   : ISO 8601 UTC timestamp string
    source        : Namespaced source label e.g. "reddit/wallstreetbets"
    content_hash  : MD5 of combined_text — used to detect duplicate content
    """
    post_id:       str
    ticker:        str
    subreddit:     str
    title:         str
    body:          str
    combined_text: str
    author:        str
    score:         int
    upvote_ratio:  float
    num_comments:  int
    url:           str
    created_utc:   str
    source:        str
    content_hash:  str


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — FETCH POSTS VIA REDDIT PUBLIC JSON
# ══════════════════════════════════════════════════════════════════════════════

def _build_session() -> requests.Session:
    """
    Build a persistent requests Session with the correct User-Agent header.

    Reddit's public JSON API requires a non-generic User-Agent string.
    Using a Session reuses the TCP connection across requests for efficiency.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _parse_post(post_data: dict, ticker: str, sub_name: str) -> RedditPost:
    """
    Parse a single post dict from Reddit's JSON response into a RedditPost.

    Parameters
    ----------
    post_data : The 'data' field of a single post object from Reddit's JSON
    ticker    : Ticker symbol being searched
    sub_name  : Subreddit name the post was fetched from
    """
    title = post_data.get("title", "") or ""
    body  = post_data.get("selftext", "") or ""

    # Discard removed/deleted body text — title may still carry sentiment signal
    if body in ("[removed]", "[deleted]"):
        body = ""

    combined     = f"{title}. {body}".strip()
    content_hash = hashlib.md5(combined.encode("utf-8")).hexdigest()

    # Convert Unix timestamp to ISO 8601
    created_utc = datetime.fromtimestamp(
        post_data.get("created_utc", 0), tz=timezone.utc
    ).isoformat()

    return RedditPost(
        post_id       = post_data.get("id", ""),
        ticker        = ticker.upper(),
        subreddit     = sub_name,
        title         = title,
        body          = body,
        combined_text = combined[:512],     # Hard cap: FinBERT token limit
        author        = post_data.get("author", "[deleted]") or "[deleted]",
        score         = post_data.get("score", 0),
        upvote_ratio  = post_data.get("upvote_ratio", 0.0),
        num_comments  = post_data.get("num_comments", 0),
        url           = f"{REDDIT_BASE_URL}{post_data.get('permalink', '')}",
        created_utc   = created_utc,
        source        = f"reddit/{sub_name}",
        content_hash  = content_hash,
    )


def fetch_subreddit_posts(
    session:     requests.Session,
    sub_name:    str,
    ticker:      str,
    post_limit:  int = 100,
    sort:        str = "hot",
    time_filter: str = "week",
) -> list[RedditPost]:
    """
    Fetch posts from a single subreddit using Reddit's public JSON endpoint.

    Paginates automatically using Reddit's 'after' cursor until post_limit
    is reached or no further results are available.

    Parameters
    ----------
    session     : Persistent requests Session with User-Agent set
    sub_name    : Subreddit name to search
    ticker      : Stock ticker symbol (e.g. "NVDA")
    post_limit  : Maximum posts to collect from this subreddit
    sort        : "hot" | "new" | "top" | "relevance"
    time_filter : "day" | "week" | "month" | "year" — only used when sort="top"
    """
    posts: list[RedditPost] = []
    after: str | None       = None
    query: str              = f"${ticker} OR {ticker} stock"

    while len(posts) < post_limit:
        params: dict = {
            "q":           query,
            "sort":        sort,
            "restrict_sr": True,
            "limit":       min(100, post_limit - len(posts)),
            "t":           time_filter if sort == "top" else "all",
        }
        if after:
            params["after"] = after

        url = f"{REDDIT_BASE_URL}/r/{sub_name}/search.json"

        try:
            response = session.get(url, params=params, timeout=10)

            # Handle rate limiting gracefully
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                log.warning(f"  Rate limited — waiting {retry_after}s before retry...")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data     = response.json()
            children = data.get("data", {}).get("children", [])

            if not children:
                break

            for child in children:
                if child.get("kind") == "t3":   # t3 = post (not comment)
                    posts.append(_parse_post(child["data"], ticker, sub_name))

            # Advance pagination cursor
            after = data.get("data", {}).get("after")
            if not after:
                break

        except requests.exceptions.RequestException as exc:
            log.warning(f"  r/{sub_name} request failed — {exc}")
            break

        time.sleep(REQUEST_DELAY)

    return posts


def fetch_posts(
    ticker:      str,
    subreddits:  list[str] = TARGET_SUBREDDITS,
    post_limit:  int       = 100,
    sort:        str       = "hot",
    time_filter: str       = "week",
) -> list[RedditPost]:
    """
    Search all target subreddits for posts mentioning the given ticker.

    No API credentials required. Builds a shared HTTP session and queries
    each subreddit sequentially with a delay to avoid rate limiting.

    Parameters
    ----------
    ticker      : Stock ticker to search for (e.g. "NVDA")
    subreddits  : List of subreddit names to search across
    post_limit  : Maximum posts to fetch per subreddit
    sort        : "hot" | "new" | "top" | "relevance"
    time_filter : Time window — only used when sort="top"
    """
    log.info(f"Fetching posts for ${ticker.upper()} — sort={sort}, limit={post_limit}/sub")
    log.info("No API credentials required — using Reddit public JSON endpoint")

    session   = _build_session()
    all_posts: list[RedditPost] = []

    for sub_name in subreddits:
        sub_posts = fetch_subreddit_posts(
            session, sub_name, ticker,
            post_limit=post_limit, sort=sort, time_filter=time_filter,
        )
        log.info(f"  r/{sub_name:<22} {len(sub_posts):>4} posts")
        all_posts.extend(sub_posts)
        time.sleep(REQUEST_DELAY)

    log.info(f"Total fetched (pre-dedup): {len(all_posts)}")
    return all_posts


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CLEAN AND DEDUPLICATE
# ══════════════════════════════════════════════════════════════════════════════

_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"http\S+"),
    re.compile(r"www\.\S+"),
    re.compile(r"\[.*?\]\(.*?\)"),
    re.compile(r"\*\*|\*|__|##|#"),
    re.compile(r"&amp;|&lt;|&gt;|&nbsp;"),
    re.compile(r"[^\x00-\x7F]+"),
    re.compile(r"\s{2,}"),
]


def clean_text(text: str) -> str:
    """Strip noise from raw Reddit text to improve NLP signal quality."""
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    return text.strip()


def deduplicate(posts: list[RedditPost]) -> list[RedditPost]:
    """
    Remove duplicate posts by post_id and content_hash.

    post_id      catches the same post returned across multiple subreddits.
    content_hash catches near-identical reposts with different metadata.
    """
    seen_ids:    set[str] = set()
    seen_hashes: set[str] = set()
    unique:      list[RedditPost] = []

    for post in posts:
        if post.post_id in seen_ids or post.content_hash in seen_hashes:
            continue
        seen_ids.add(post.post_id)
        seen_hashes.add(post.content_hash)
        post.combined_text = clean_text(post.combined_text)
        unique.append(post)

    removed = len(posts) - len(unique)
    log.info(f"Deduplication: {removed} duplicates removed → {len(unique)} unique posts")
    return unique


def filter_quality(
    posts:           list[RedditPost],
    min_text_length: int = MIN_TEXT_LENGTH,
) -> list[RedditPost]:
    """Drop posts too short to carry meaningful sentiment signal."""
    filtered = [p for p in posts if len(p.combined_text) >= min_text_length]
    removed  = len(posts) - len(filtered)
    log.info(f"Quality filter: {removed} short posts removed → {len(filtered)} posts remain")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — METADATA SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def summarise_metadata(posts: list[RedditPost], ticker: str) -> dict:
    """
    Compute aggregate metadata statistics across all collected posts.

    Contextualises the sentiment score — a bearish result on 10 posts is
    far less meaningful than the same score on 400 posts.
    """
    if not posts:
        return {}

    scores   = [p.score        for p in posts]
    comments = [p.num_comments for p in posts]
    sub_dist = pd.Series([p.subreddit for p in posts]).value_counts().to_dict()

    metadata = {
        "ticker":             ticker.upper(),
        "total_posts":        len(posts),
        "subreddits_covered": sub_dist,
        "avg_upvote_score":   round(sum(scores)   / len(scores),   2),
        "max_upvote_score":   max(scores),
        "avg_comments":       round(sum(comments) / len(comments), 2),
        "date_range_start":   min(p.created_utc for p in posts),
        "date_range_end":     max(p.created_utc for p in posts),
        "fetched_at":         datetime.now(tz=timezone.utc).isoformat(),
    }

    log.info(f"Metadata: {len(posts)} posts across {len(sub_dist)} subreddits")
    return metadata


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FINBERT SENTIMENT INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def load_finbert(model_path: str = FINBERT_MODEL, device: int = DEVICE) -> Pipeline:
    """
    Load FinBERT from Hugging Face Hub.

    FinBERT is BERT fine-tuned on financial corpora, classifying text into
    positive | neutral | negative. Model weights (~440MB) are cached after
    first download. Pass a local path to use a fine-tuned checkpoint.
    """
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return _FINBERT_PIPE

    device_label = f"GPU (device {device})" if device >= 0 else "CPU"
    log.info(f"Loading FinBERT from '{model_path}' on {device_label}...")

    _FINBERT_PIPE = pipeline(
        task="text-classification",
        model=model_path,
        tokenizer=model_path,
        device=device,
        truncation=True,
        max_length=512,
    )

    log.info("FinBERT loaded and ready")
    return _FINBERT_PIPE


def apply_sentiment(
    posts:        list[RedditPost],
    finbert_pipe: Pipeline,
    batch_size:   int = BATCH_SIZE,
) -> list[dict]:
    """
    Run batched FinBERT inference and attach sentiment fields to each post.

    Each post receives:
      sentiment_label    — "positive" | "neutral" | "negative"
      sentiment_score    — 1 | 0 | -1
      confidence         — FinBERT probability for the winning label
      weighted_sentiment — sentiment_score x confidence

    weighted_sentiment is the primary signal fed into the composite score.
    It encodes both direction and model certainty in a single value.
    """
    log.info(f"Running FinBERT on {len(posts)} posts (batch_size={batch_size})...")

    texts       = [p.combined_text for p in posts]
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT inference"):
        batch = texts[i : i + batch_size]
        predictions.extend(finbert_pipe(batch))

    enriched: list[dict] = []
    for post, pred in zip(posts, predictions):
        label      = pred["label"].lower()
        confidence = round(pred["score"], 4)
        num_score  = SENTIMENT_SCORE_MAP.get(label, 0)
        weighted   = round(num_score * confidence, 4)

        record = asdict(post)
        record.update({
            "sentiment_label":    label,
            "sentiment_score":    num_score,
            "confidence":         confidence,
            "weighted_sentiment": weighted,
        })
        enriched.append(record)

    log.info("Inference complete")
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — OUTPUT STRUCTURED DATASET
# ══════════════════════════════════════════════════════════════════════════════

_CSV_COLUMNS: list[str] = [
    "ticker", "post_id", "subreddit", "source",
    "created_utc", "author",
    "title", "body", "combined_text",
    "score", "upvote_ratio", "num_comments",
    "sentiment_label", "sentiment_score", "confidence", "weighted_sentiment",
    "url", "content_hash",
]


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    """Convert enriched post records into a tidy DataFrame sorted by date descending."""
    df = pd.DataFrame(records)
    df = df[[c for c in _CSV_COLUMNS if c in df.columns]]
    df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
    df = df.sort_values("created_utc", ascending=False).reset_index(drop=True)
    return df


def compute_aggregate_score(df: pd.DataFrame) -> dict:
    """
    Compute a composite sentiment score using upvote-weighted averaging.

    Posts with more upvotes carry proportionally more weight, reflecting
    genuine community conviction over low-engagement noise.
    Scores clipped to minimum 1 to prevent division-by-zero.
    """
    weight_col   = df["score"].clip(lower=1)
    total_weight = weight_col.sum()
    weighted_sum = (df["weighted_sentiment"] * weight_col).sum()

    return {
        "ticker":               df["ticker"].iloc[0],
        "composite_sentiment":  round(weighted_sum / total_weight, 4),
        "avg_sentiment":        round(df["weighted_sentiment"].mean(), 4),
        "bullish_pct":          round((df["sentiment_label"] == "positive").mean(), 4),
        "neutral_pct":          round((df["sentiment_label"] == "neutral").mean(), 4),
        "bearish_pct":          round((df["sentiment_label"] == "negative").mean(), 4),
        "total_posts_analysed": len(df),
        "avg_confidence":       round(df["confidence"].mean(), 4),
    }


def score_to_outlook_label(score: float) -> str:
    if score >= 0.6:
        return "Bullish"
    if score <= 0.4:
        return "Bearish"
    return "Neutral"


def build_social_summary(df: pd.DataFrame) -> dict:
    if df.empty:
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

    aggregate = compute_aggregate_score(df)
    counts = df["sentiment_label"].str.lower().value_counts()
    overall_score = float(aggregate["composite_sentiment"])

    return {
        "article_count": int(len(df)),
        "average_sentiment_score": round(float(df["sentiment_score"].mean()), 4),
        "weighted_sentiment_score": round(float(df["weighted_sentiment"].mean()), 4),
        "overall_sentiment_score": round(overall_score, 4),
        "overall_sentiment_label": score_to_outlook_label(overall_score),
        "positive_articles": int(counts.get("positive", 0)),
        "negative_articles": int(counts.get("negative", 0)),
        "neutral_articles": int(counts.get("neutral", 0)),
    }


def build_social_overall_summary(df: pd.DataFrame, sentiment_summary: dict) -> str:
    if df.empty:
        return "No usable social posts were found for this ticker, so there is not enough social sentiment context yet."

    top_subreddits = df["subreddit"].fillna("Unknown").value_counts().head(3).index.tolist()
    top_titles = df["title"].fillna("").head(2).tolist()
    tone = sentiment_summary["overall_sentiment_label"].lower()
    score = sentiment_summary["overall_sentiment_score"]
    item_count = sentiment_summary["article_count"]

    subreddit_text = ", ".join(top_subreddits) if top_subreddits else "tracked subreddits"
    title_text = " | ".join([title for title in top_titles if title]) or "No standout post captured"

    return (
        f"Across {item_count} recent Reddit posts, the social sentiment looks {tone} "
        f"with an aggregate score of {score} on a -1 to 1 scale, where 0 is neutral. "
        f"The strongest discussion activity came from {subreddit_text}. "
        f"Recent posts focused on: {title_text}."
    )


def normalize_social_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "ticker", "title", "source", "published", "summary", "url",
                "sentiment_label", "sentiment_score", "score", "num_comments", "subreddit",
            ]
        )

    normalized = df.copy()
    normalized["published"] = pd.to_datetime(normalized["created_utc"], utc=True, errors="coerce")
    normalized["summary"] = normalized["combined_text"].fillna("").str.slice(0, 240)
    normalized["source"] = normalized["source"].fillna("reddit")
    normalized = normalized.sort_values("published", ascending=False).reset_index(drop=True)
    return normalized


def analyze_social_sentiment(
    ticker: str,
    post_limit: int = 25,
    sort: str = "hot",
    time_filter: str = "week",
    finbert_model: str = FINBERT_MODEL,
    device: int = DEVICE,
) -> dict:
    ticker = ticker.upper()

    raw_posts = fetch_posts(
        ticker,
        subreddits=TARGET_SUBREDDITS,
        post_limit=post_limit,
        sort=sort,
        time_filter=time_filter,
    )
    posts = filter_quality(deduplicate(raw_posts))

    if not posts:
        empty_df = normalize_social_dataframe(pd.DataFrame())
        empty_summary = build_social_summary(empty_df)
        return {
            "social_df": empty_df,
            "social_sentiment_summary": empty_summary,
            "social_overall_summary": build_social_overall_summary(empty_df, empty_summary),
            "social_metadata": {},
        }

    metadata = summarise_metadata(posts, ticker)
    finbert = load_finbert(finbert_model, device=device)
    records = apply_sentiment(posts, finbert)
    social_df = normalize_social_dataframe(build_dataframe(records))
    sentiment_summary = build_social_summary(social_df)

    return {
        "social_df": social_df,
        "social_sentiment_summary": sentiment_summary,
        "social_overall_summary": build_social_overall_summary(social_df, sentiment_summary),
        "social_metadata": metadata,
    }


def save_outputs(
    df: pd.DataFrame, metadata: dict, aggregate: dict, ticker: str
) -> tuple[str, str]:
    """Save timestamped CSV and JSON summary files to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = f"{ticker}_sentiment_{timestamp}.csv"
    json_path = f"{ticker}_summary_{timestamp}.json"

    df.to_csv(csv_path, index=False)
    log.info(f"Dataset saved  → {csv_path}  ({len(df)} rows)")

    summary = {"metadata": metadata, "aggregate_sentiment": aggregate}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved  → {json_path}")

    return csv_path, json_path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    ticker:        str,
    post_limit:    int = 100,
    sort:          str = "hot",
    time_filter:   str = "week",
    finbert_model: str = FINBERT_MODEL,
    device:        int = DEVICE,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the full end-to-end sentiment pipeline for a single ticker.
    No API credentials required.

    Parameters
    ----------
    ticker        : Stock ticker symbol (e.g. "NVDA", "AAPL", "TSLA")
    post_limit    : Max posts per subreddit (default 100)
    sort          : "hot" | "new" | "top" | "relevance"
    time_filter   : "day" | "week" | "month" | "year" — only used when sort="top"
    finbert_model : Hugging Face model ID or local fine-tuned checkpoint path
    device        : 0 = GPU, -1 = CPU (auto-detected by default)

    Returns
    -------
    df      — pd.DataFrame, one row per post with all sentiment columns
    summary — dict with metadata and aggregate sentiment score
    """
    ticker = ticker.upper()
    log.info(f"{'═' * 60}")
    log.info(f"  Pipeline start — ${ticker}")
    log.info(f"  Device: {'GPU' if device >= 0 else 'CPU'} | Sort: {sort} | Limit: {post_limit}/sub")
    log.info(f"{'═' * 60}")

    # Step 1: Fetch
    raw_posts = fetch_posts(
        ticker, subreddits=TARGET_SUBREDDITS,
        post_limit=post_limit, sort=sort, time_filter=time_filter,
    )

    # Step 2: Clean & Deduplicate
    posts = deduplicate(raw_posts)
    posts = filter_quality(posts)

    if not posts:
        log.warning("No usable posts found after filtering. Pipeline halted.")
        return pd.DataFrame(), {}

    # Step 3: Metadata
    metadata = summarise_metadata(posts, ticker)

    # Step 4: FinBERT Inference
    finbert = load_finbert(finbert_model, device=device)
    records = apply_sentiment(posts, finbert)

    # Step 5: Build & Save Output
    df        = build_dataframe(records)
    aggregate = compute_aggregate_score(df)
    save_outputs(df, metadata, aggregate, ticker)

    log.info(f"{'─' * 60}")
    log.info(f"  Aggregate Sentiment — ${ticker}")
    log.info(f"{'─' * 60}")
    for key, val in aggregate.items():
        log.info(f"  {key:<28} {val}")
    log.info(f"{'─' * 60}")

    return df, {"metadata": metadata, "aggregate_sentiment": aggregate}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Single ticker
    df, summary = run_pipeline(
        ticker      = "NVDA",
        post_limit  = 100,
        sort        = "hot",
        time_filter = "week",
    )

    if not df.empty:
        print("\n── Sample Output (top 10 rows) ──")
        print(df[[
            "ticker", "subreddit", "title",
            "sentiment_label", "confidence", "weighted_sentiment"
        ]].head(10).to_string(index=False))

    # Multiple tickers (uncomment to use)
    # for ticker in ["AAPL", "TSLA", "NVDA", "AMZN"]:
    #     run_pipeline(ticker, post_limit=100, sort="hot")
