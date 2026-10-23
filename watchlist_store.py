"""
User-added watchlist tickers, layered on top of the hardcoded
config.WATCHLIST_SECTIONS. Lets someone add a stock from the running app
(My Watchlist page) without anyone editing code.

The deployed app (Streamlit Community Cloud) has no persistent disk --
anything written to the local filesystem is lost whenever the container
restarts or redeploys. So an addition isn't considered saved until it's
committed back to GitHub via the REST API; every page that reads the
watchlist re-reads WATCHLIST_ADDITIONS_PATH, so once the commit lands and
Streamlit Cloud auto-redeploys, the new ticker shows up everywhere.

Needs a GitHub token to commit as the deployed app:
  - Streamlit Cloud: add a secret named GITHUB_TOKEN (Settings -> Secrets)
    using a fine-grained personal access token scoped to just this repo,
    with Contents: Read and write permission.
  - Running locally: falls back to the GITHUB_TOKEN env var, then to
    `gh auth token` if the GitHub CLI is authenticated.
If no token is available anywhere, add_ticker() still saves locally (so it
works immediately in a local session) but reports that it won't survive a
redeploy until it's pushed some other way.
"""
import json
from pathlib import Path

from classification import classify_section
from config import WATCHLIST_SECTIONS
from data_fetcher import fetch_fundamentals, validate_tickers
from github_sync import commit_json, get_token

ADDITIONS_PATH = Path(__file__).parent / "watchlist_additions.json"

GITHUB_REPO = "jxyminn/stock_ranker"
GITHUB_BRANCH = "main"
GITHUB_FILE_PATH = "watchlist_additions.json"


def _load_additions() -> dict[str, list[str]]:
    if not ADDITIONS_PATH.exists():
        return {}
    try:
        return json.loads(ADDITIONS_PATH.read_text())
    except Exception:
        return {}


def get_effective_sections() -> dict[str, list[str]]:
    """config.WATCHLIST_SECTIONS plus any user-added tickers, merged."""
    merged = {k: list(v) for k, v in WATCHLIST_SECTIONS.items()}
    for section, tickers in _load_additions().items():
        merged.setdefault(section, [])
        for t in tickers:
            if t not in merged[section]:
                merged[section].append(t)
    return merged


def ticker_exists(ticker: str) -> bool:
    ticker = ticker.strip().upper()
    for tickers in get_effective_sections().values():
        if ticker in tickers:
            return True
    return False


def add_ticker(query: str) -> dict:
    """
    Resolve, classify, and persist a new ticker. Returns:
      {"ok": bool, "ticker": str|None, "section": str|None, "message": str, "persisted": bool}
    "persisted" is True only if the change actually made it to GitHub --
    False means it's local-only (or in-memory-only) and will NOT survive a
    redeploy of the hosted app.
    """
    from data_fetcher import search_ticker

    candidate = query.strip().upper()
    if not validate_tickers([candidate]):
        matches = search_ticker(query)
        equity_matches = [m for m in matches if m.get("symbol")]
        if not equity_matches:
            return {"ok": False, "ticker": None, "section": None,
                     "message": f"No matching stock found for '{query}'.", "persisted": False}
        candidate = equity_matches[0]["symbol"].upper()

    if ticker_exists(candidate):
        return {"ok": False, "ticker": candidate, "section": None,
                 "message": f"{candidate} is already in the watchlist.", "persisted": False}

    fund = fetch_fundamentals(candidate)
    section = classify_section(fund)

    additions = _load_additions()
    additions.setdefault(section, [])
    additions[section].append(candidate)
    ADDITIONS_PATH.write_text(json.dumps(additions, indent=2) + "\n")

    token = get_token()
    if token:
        ok, detail = commit_json(GITHUB_REPO, GITHUB_BRANCH, GITHUB_FILE_PATH, additions,
                                  token, f"Add {candidate} to watchlist (via app)")
        if ok:
            return {"ok": True, "ticker": candidate, "section": section,
                     "message": f"Added {candidate} to \"{section}\" and saved permanently.",
                     "persisted": True}
        return {"ok": True, "ticker": candidate, "section": section,
                 "message": f"Added {candidate} to \"{section}\" locally, but couldn't save to "
                            f"GitHub ({detail}). It'll be lost if this app restarts.",
                 "persisted": False}

    return {"ok": True, "ticker": candidate, "section": section,
             "message": f"Added {candidate} to \"{section}\" locally. No GitHub token is "
                        f"configured, so this won't survive a redeploy -- see watchlist_store.py's "
                        f"docstring for setup.",
             "persisted": False}
