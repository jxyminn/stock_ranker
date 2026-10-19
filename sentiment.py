"""
Headline sentiment scoring for analyze.py's "Recent headlines" section.
Report-only: this is informational context for the human reading the report,
never a feature fed into the trained models (see fetch_news's docstring for
why -- no historical archive means no way to backfill this into training
data without lookahead).

Uses VADER (a lexicon + rule-based sentiment scorer tuned for short,
informal text) as the base engine, extended with finance-specific
vocabulary VADER's general-purpose lexicon misses ("soars", "downgrade",
"beat estimates", etc. score as neutral otherwise).
"""
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER lexicon scores run roughly -4 (most negative) to +4 (most positive).
# This list is deliberately modest -- common financial-headline vocabulary,
# not an attempt at a comprehensive domain lexicon.
FINANCE_LEXICON = {
    # positive
    "soar": 3.0, "soars": 3.0, "soaring": 3.0,
    "surge": 2.8, "surges": 2.8, "surging": 2.8,
    "rally": 2.5, "rallies": 2.5, "rallying": 2.5,
    "beat": 2.0, "beats": 2.0, "beating": 2.0,
    "outperform": 2.2, "outperforms": 2.2, "outperformed": 2.2,
    "upgrade": 2.3, "upgrades": 2.3, "upgraded": 2.3,
    "record high": 2.5, "all-time high": 2.5,
    "blowout": 2.5, "bullish": 2.5,
    "raises guidance": 2.2, "raised guidance": 2.2,
    "buyback": 1.5, "expansion": 1.2,
    # negative
    "plunge": -3.0, "plunges": -3.0, "plunging": -3.0,
    "slump": -2.5, "slumps": -2.5, "slumping": -2.5,
    "tumble": -2.6, "tumbles": -2.6, "tumbling": -2.6,
    "miss": -2.0, "misses": -2.0, "missed": -2.0,
    "downgrade": -2.3, "downgrades": -2.3, "downgraded": -2.3,
    "recall": -2.0, "recalls": -2.0,
    "lawsuit": -2.2, "sues": -2.2, "sued": -2.2,
    "investigation": -2.0, "probe": -1.8, "subpoena": -2.0,
    "bearish": -2.5, "layoffs": -2.2, "layoff": -2.2,
    "cuts guidance": -2.2, "cut guidance": -2.2, "lowers guidance": -2.2,
    "bankruptcy": -3.5, "fraud": -3.0, "scandal": -2.8,
    "warns": -1.5, "warning": -1.5, "delisted": -3.0,
}


@lru_cache(maxsize=1)
def _analyzer() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FINANCE_LEXICON)
    return sia


def score_headline(text: str) -> float:
    """Compound sentiment score in [-1, 1] (negative -> positive)."""
    if not text:
        return 0.0
    return _analyzer().polarity_scores(text)["compound"]


def label_for_score(score: float) -> str:
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def summarize_articles(articles: list[dict]) -> dict:
    """
    articles: list of {title, publisher, link, published} dicts (see
    data_fetcher.fetch_news). Returns per-article scores plus an aggregate.
    """
    scored = []
    for a in articles:
        score = score_headline(a.get("title", ""))
        scored.append({**a, "sentiment_score": score, "sentiment_label": label_for_score(score)})

    if not scored:
        return {"articles": [], "mean_score": None, "overall_label": "no recent headlines"}

    mean_score = sum(a["sentiment_score"] for a in scored) / len(scored)
    return {
        "articles": scored,
        "mean_score": mean_score,
        "overall_label": label_for_score(mean_score),
    }
