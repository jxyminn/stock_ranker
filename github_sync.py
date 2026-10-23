"""
Shared helper for committing a JSON file back to a GitHub repo via the
Contents API. Used by watchlist_store.py (public stock_ranker repo) and
portfolio_store.py (private stock_ranker_portfolio repo) so a user can add
data from the running app itself and have it survive a redeploy of the
otherwise-stateless deployed Streamlit app.
"""
import base64
import json
import subprocess

import requests


def get_token() -> str | None:
    """
    Token lookup order:
      1. Streamlit secret GITHUB_TOKEN (how the deployed app gets it)
      2. GITHUB_TOKEN environment variable
      3. `gh auth token` (convenience for local sessions already logged in)
    """
    try:
        import streamlit as st
        token = st.secrets.get("GITHUB_TOKEN")
        if token:
            return token
    except Exception:
        pass

    import os
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ["GITHUB_TOKEN"]

    try:
        result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass

    return None


def commit_json(repo: str, branch: str, file_path: str, content: dict,
                 token: str, commit_message: str) -> tuple[bool, str]:
    """Write `content` as JSON to `file_path` in `repo` via the Contents API."""
    api_url = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    sha = None
    try:
        resp = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=15)
        if resp.status_code == 200:
            sha = resp.json()["sha"]
        elif resp.status_code != 404:
            return False, f"GitHub API error checking existing file: {resp.status_code} {resp.text[:200]}"
    except requests.RequestException as e:
        return False, f"Couldn't reach GitHub API: {e}"

    body_text = json.dumps(content, indent=2) + "\n"
    payload = {
        "message": commit_message,
        "content": base64.b64encode(body_text.encode()).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    try:
        resp = requests.put(api_url, headers=headers, json=payload, timeout=15)
    except requests.RequestException as e:
        return False, f"Couldn't reach GitHub API: {e}"

    if resp.status_code in (200, 201):
        return True, "committed"
    return False, f"GitHub API error committing: {resp.status_code} {resp.text[:300]}"
