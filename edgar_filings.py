import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# =============================
# CONFIG
# =============================

# Replace with your real name + email for SEC requests
USER_AGENT = "Robert Czarnota robertwc123@gmail.com"

HEADERS_DATA = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

HEADERS_ARCHIVE = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

TARGET_FORMS = {"8-K", "10-Q", "10-K"}
FINBERT_BATCH_SIZE = int(os.getenv("FINBERT_BATCH_SIZE", "16"))

EVENT_KEYWORDS = {
    "earnings": [
        "financial results", "quarterly results", "annual results",
        "results of operations", "earnings"
    ],
    "guidance": [
        "guidance", "outlook", "forecast"
    ],
    "leadership_change": [
        "chief executive officer", "chief financial officer",
        "resigned", "appointed", "director"
    ],
    "acquisition": [
        "acquisition", "acquired", "merger", "definitive agreement"
    ],
    "investigation": [
        "investigation", "subpoena", "department of justice",
        "sec inquiry", "sec investigation"
    ],
    "bankruptcy": [
        "bankruptcy", "chapter 11", "restructuring"
    ],
}

# =============================
# HUGGING FACE TOKEN CHECK
# =============================

HF_TOKEN = os.getenv("HF_TOKEN")
finbert = None

if HF_TOKEN:
    print("HF_TOKEN found. Using authenticated Hugging Face requests.")
else:
    print("Warning: HF_TOKEN not found.")
    print("The model may still load, but downloads can be slower or hit rate limits.")
    print('In PowerShell, set it with: $env:HF_TOKEN="hf_your_token_here"')

# =============================
# LOAD FINBERT
# =============================

def get_finbert_pipeline():
    global finbert

    if finbert is None:
        finbert = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            token=HF_TOKEN if HF_TOKEN else None,
        )
        print("FinBERT loaded successfully")

    return finbert

# =============================
# TEXT HELPERS
# =============================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    try:
        return text.encode("latin1").decode("utf-8")
    except Exception:
        return text


def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return clean_text(text)


def normalize_accession(accession):
    return accession.replace("-", "")


def safe_filename_part(text):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text))


# =============================
# SENTIMENT + EVENT TAGGING
# =============================

def get_finbert_sentiment(text):
    if not text:
        return None, None

    try:
        result = get_finbert_pipeline()(text[:512])[0]
        return result["label"], result["score"]
    except Exception as e:
        print(f"Sentiment error: {e}")
        return None, None


def get_finbert_sentiments(texts, batch_size=FINBERT_BATCH_SIZE):
    prepared_texts = [(text or "")[:512] for text in texts]
    valid_texts = [text for text in prepared_texts if text]

    try:
        predictions = iter(get_finbert_pipeline()(valid_texts, batch_size=batch_size)) if valid_texts else iter([])
    except Exception as e:
        print(f"Sentiment batch error: {e}")
        return [(None, None) for _ in prepared_texts]

    results = []
    for text in prepared_texts:
        if not text:
            results.append((None, None))
            continue

        result = next(predictions)
        results.append((result["label"], result["score"]))

    return results


def detect_event_tags(text):
    if not text:
        return []

    text_lower = text.lower()
    found_tags = []

    for tag, keywords in EVENT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            found_tags.append(tag)

    return found_tags


def describe_form_type(form):
    form = str(form).upper()
    if form == "10-K":
        return "annual report"
    if form == "10-Q":
        return "quarterly report"
    if form == "8-K":
        return "current report about a recent company event"
    return "SEC filing"


def format_event_tag(tag):
    return tag.replace("_", " ")


def is_readable_filing_sentence(sentence):
    sentence = clean_text(sentence)
    if len(sentence) < 45 or len(sentence) > 300:
        return False

    lower_sentence = sentence.lower()
    noisy_terms = [
        "xbrl", "xbrli", "iso4217", "dei:", "aapl:", "0000", "contextref",
        "unitref", "decimals", "p1y", "p329d", "member]", "duration_",
    ]
    if any(term in lower_sentence for term in noisy_terms):
        return False

    tokens = sentence.split()
    if len(tokens) < 8:
        return False
    if any(len(token) > 38 for token in tokens):
        return False

    alpha_chars = sum(1 for char in sentence if char.isalpha())
    visible_chars = sum(1 for char in sentence if not char.isspace())
    if visible_chars == 0:
        return False

    return alpha_chars / visible_chars >= 0.55


def extract_plain_filing_excerpt(text, max_sentences=3):
    if not text:
        return ""

    important_terms = [
        "revenue", "net sales", "income", "earnings", "cash", "margin",
        "risk", "results", "operations", "guidance", "agreement",
        "appointed", "resigned", "acquisition", "litigation",
    ]
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    readable_sentences = [
        sentence.strip()
        for sentence in sentences
        if is_readable_filing_sentence(sentence)
    ]
    priority_sentences = [
        sentence
        for sentence in readable_sentences
        if any(term in sentence.lower() for term in important_terms)
    ]
    selected = (priority_sentences or readable_sentences)[:max_sentences]
    return " ".join(selected)


def build_plain_language_filing_summary(company_name, form, filing_date, description, event_tags, text):
    form_description = describe_form_type(form)
    filing_date_text = pd.Timestamp(filing_date).date().isoformat()
    summary_parts = [
        f"{company_name} filed a {form} {form_description} with the SEC on {filing_date_text}."
    ]

    if description:
        summary_parts.append(f"The filing description is: {description}.")

    if event_tags:
        tags_text = ", ".join(format_event_tag(tag) for tag in event_tags)
        summary_parts.append(f"Detected themes include {tags_text}.")
    else:
        summary_parts.append(
            "No specific event theme was automatically detected, so this should be read as general filing context."
        )

    excerpt = extract_plain_filing_excerpt(text)
    if excerpt:
        summary_parts.append(f"Plain-language excerpt: {excerpt}")

    return " ".join(summary_parts)


# =============================
# SEC LOOKUP FUNCTIONS
# =============================

def get_ticker_to_cik_map():
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=HEADERS_ARCHIVE, timeout=30)
    response.raise_for_status()

    data = response.json()
    ticker_map = {}

    for _, item in data.items():
        ticker = item["ticker"].upper()
        cik = str(item["cik_str"]).zfill(10)
        company = item["title"]
        ticker_map[ticker] = {
            "cik": cik,
            "company": company
        }

    return ticker_map


def get_cik_from_ticker(ticker):
    ticker = ticker.upper()
    ticker_map = get_ticker_to_cik_map()

    if ticker not in ticker_map:
        raise ValueError(f"Ticker '{ticker}' not found in SEC ticker list.")

    return ticker_map[ticker]["cik"], ticker_map[ticker]["company"]


def get_recent_filings(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS_DATA, timeout=30)
    response.raise_for_status()

    data = response.json()
    recent = data.get("filings", {}).get("recent", {})

    filings_df = pd.DataFrame({
        "form": recent.get("form", []),
        "filingDate": recent.get("filingDate", []),
        "accessionNumber": recent.get("accessionNumber", []),
        "primaryDocument": recent.get("primaryDocument", []),
        "primaryDocDescription": recent.get("primaryDocDescription", [])
    })

    return filings_df


def filter_filings_by_forms_and_date(filings_df, days_back=30):
    if filings_df.empty:
        return filings_df

    temp_df = filings_df.copy()
    temp_df["form"] = temp_df["form"].astype(str).str.upper()
    temp_df["filingDate"] = pd.to_datetime(temp_df["filingDate"], errors="coerce")

    cutoff_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days_back)

    out = temp_df[
        temp_df["form"].isin(TARGET_FORMS) &
        (temp_df["filingDate"] >= cutoff_date)
    ].copy()

    out = out.sort_values("filingDate", ascending=False)
    return out


def build_filing_url(cik, accession_number, primary_document):
    accession_nodash = normalize_accession(accession_number)
    cik_no_leading_zeros = str(int(cik))

    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_nodash}/{primary_document}"
    )


def download_filing_text(url):
    response = requests.get(url, headers=HEADERS_ARCHIVE, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    text_preview = response.text[:500].lower()

    if "html" in content_type or "<html" in text_preview:
        return extract_text_from_html(response.text)

    return clean_text(response.text)


# =============================
# MAIN DATASET BUILDER
# =============================

def build_edgar_dataset(ticker, days_back=30, delay=0.2, max_filings=None):
    print(f"\nLooking up company info for {ticker}...")
    cik, company_name = get_cik_from_ticker(ticker)
    print(f"Company: {company_name}")
    print(f"CIK: {cik}")

    print("\nFetching recent filings from SEC...")
    filings_df = get_recent_filings(cik)
    print(f"Recent filings pulled: {len(filings_df)}")

    target_filings = filter_filings_by_forms_and_date(
        filings_df,
        days_back=days_back
    )

    if max_filings is not None:
        target_filings = target_filings.head(max_filings).copy()

    print(f"Matching 8-K / 10-Q / 10-K filings in last {days_back} days: {len(target_filings)}")

    if not target_filings.empty:
        print(target_filings[["form", "filingDate"]].head())

    data = []
    sentiment_inputs = []

    for i, row in enumerate(target_filings.itertuples(index=False), start=1):
        filing_url = build_filing_url(cik, row.accessionNumber, row.primaryDocument)
        title = f"{ticker.upper()} {row.form} filing"

        print(f"[{i}/{len(target_filings)}] Downloading {row.form}: {filing_url}")

        try:
            text = download_filing_text(filing_url)
        except Exception as e:
            print(f"Failed to download filing: {e}")
            continue

        if not text or len(text) < 500:
            print("Skipped: filing text too short")
            continue

        event_tags = detect_event_tags(text)
        summary = build_plain_language_filing_summary(
            company_name=company_name,
            form=row.form,
            filing_date=row.filingDate,
            description=row.primaryDocDescription,
            event_tags=event_tags,
            text=text,
        )
        sentiment_input = extract_plain_filing_excerpt(text, max_sentences=6) or summary
        sentiment_inputs.append(sentiment_input)

        data.append({
            "ticker": ticker.upper(),
            "company": company_name,
            "source": "SEC EDGAR",
            "published": row.filingDate,
            "title": title,
            "url": filing_url,
            "summary": summary,
            "text": text,
            "text_length": len(text),

            # EDGAR-specific fields
            "form": row.form,
            "filing_date": row.filingDate,
            "accession_number": row.accessionNumber,
            "primary_document": row.primaryDocument,
            "primary_doc_description": row.primaryDocDescription,
            "event_tags": ", ".join(event_tags) if event_tags else "",
            "days_back_window": days_back,
            "is_official_sec_event": True
        })

        time.sleep(delay)

    for row, (sentiment_label, sentiment_confidence) in zip(data, get_finbert_sentiments(sentiment_inputs)):
        row["sentiment_label"] = sentiment_label
        row["sentiment_confidence"] = sentiment_confidence

    df = pd.DataFrame(data)
    print(f"\nFinal dataset size: {len(df)} filings")
    return df


# =============================
# RUN SCRIPT
# =============================

if __name__ == "__main__":
    ticker = input("Enter ticker symbol (e.g., AAPL): ").upper().strip()

    try:
        days_back = int(input("How many days back should we check SEC filings? ").strip())
    except Exception:
        days_back = 30

    df = build_edgar_dataset(
        ticker=ticker,
        days_back=days_back
    )

    print("\nSample output:")
    print(df.head())

    output_name = f"edgar_{safe_filename_part(ticker)}_last_{days_back}_days.csv"
    df.to_csv(output_name, index=False)
    print(f"\nSaved to {output_name}")
