"""
Manual portfolio tracking: positions the user actually owns (ticker, shares,
cost basis), entered by hand. No brokerage API integration on purpose --
every option either costs money or requires opening/connecting a specific
account, which isn't worth it for a hobby tool.

Stored in a SEPARATE PRIVATE repo (jxyminn/stock_ranker_portfolio), not the
public stock_ranker repo -- unlike watchlist tickers, share counts and cost
basis reveal actual dollar amounts, which shouldn't be public. Uses the same
GitHub Contents API commit pattern as watchlist_store.py (see github_sync.py)
so it persists across redeploys of the otherwise-stateless deployed app.

Needs the same GITHUB_TOKEN as watchlist_store.py, but that token must also
have access to the PRIVATE stock_ranker_portfolio repo -- if the token is a
fine-grained personal access token scoped to just the public repo, its
repository access needs to be edited (GitHub Settings -> Developer settings
-> Fine-grained tokens) to include stock_ranker_portfolio too.
"""
import base64
import json
import uuid
from pathlib import Path

import requests

from data_fetcher import validate_tickers
from github_sync import commit_json, get_token

PORTFOLIO_PATH = Path(__file__).parent / "portfolio.json"  # local cache/fallback only

GITHUB_REPO = "jxyminn/stock_ranker_portfolio"
GITHUB_BRANCH = "main"
GITHUB_FILE_PATH = "portfolio.json"


def _load_local() -> dict:
    if not PORTFOLIO_PATH.exists():
        return {"positions": []}
    try:
        return json.loads(PORTFOLIO_PATH.read_text())
    except Exception:
        return {"positions": []}


def _fetch_remote() -> dict | None:
    """
    Pull portfolio.json straight from the private repo via the Contents API
    -- the deployed app's local disk doesn't persist across restarts, so the
    private repo (not whatever's cached in this container) is the actual
    source of truth. Deliberately NOT using raw.githubusercontent.com: that
    URL is served through a CDN that can lag behind a just-made commit by
    seconds to minutes, which would make an add-then-immediately-reload
    appear to silently lose the new position. The Contents API reflects a
    commit immediately.
    """
    token = get_token()
    if not token:
        return None
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}",
            headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            params={"ref": GITHUB_BRANCH}, timeout=15,
        )
        if resp.status_code == 200:
            return json.loads(base64.b64decode(resp.json()["content"]))
    except Exception:
        pass
    return None


def load_portfolio() -> dict:
    """Prefer the private repo's live copy; fall back to the local cache
    (no token configured, or a transient network error)."""
    remote = _fetch_remote()
    if remote is not None:
        PORTFOLIO_PATH.write_text(json.dumps(remote, indent=2) + "\n")
        return remote
    return _load_local()


def _save(portfolio: dict, message: str) -> tuple[bool, str]:
    PORTFOLIO_PATH.write_text(json.dumps(portfolio, indent=2) + "\n")
    token = get_token()
    if not token:
        return False, "No GitHub token configured -- saved locally only, won't survive a redeploy."
    ok, detail = commit_json(GITHUB_REPO, GITHUB_BRANCH, GITHUB_FILE_PATH, portfolio, token, message)
    if ok:
        return True, "Saved."
    return False, f"Couldn't save to GitHub ({detail}). Saved locally only."


def add_position(ticker: str, shares: float, cost_basis_per_share: float,
                  purchase_date: str | None = None, notes: str = "") -> dict:
    ticker = ticker.strip().upper()
    if not validate_tickers([ticker]):
        return {"ok": False, "message": f"'{ticker}' doesn't look like a valid ticker."}
    if shares <= 0:
        return {"ok": False, "message": "Shares must be greater than 0."}
    if cost_basis_per_share < 0:
        return {"ok": False, "message": "Cost basis can't be negative."}

    portfolio = load_portfolio()
    portfolio.setdefault("positions", []).append({
        "id": uuid.uuid4().hex[:8],
        "ticker": ticker,
        "shares": shares,
        "cost_basis_per_share": cost_basis_per_share,
        "purchase_date": purchase_date,
        "notes": notes,
    })
    persisted, message = _save(portfolio, f"Add {ticker} position (via app)")
    return {"ok": True, "persisted": persisted, "message": message}


def remove_position(position_id: str) -> dict:
    portfolio = load_portfolio()
    before = len(portfolio.get("positions", []))
    portfolio["positions"] = [p for p in portfolio.get("positions", []) if p["id"] != position_id]
    if len(portfolio["positions"]) == before:
        return {"ok": False, "message": "Position not found."}
    persisted, message = _save(portfolio, "Remove position (via app)")
    return {"ok": True, "persisted": persisted, "message": message}
