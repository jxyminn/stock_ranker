"""
Look up a ticker symbol by company name (or partial symbol) before adding it
to your watchlist / tickers file.

Usage:
    python search.py "palantir"
    python search.py "advanced micro"
"""
import sys
from data_fetcher import search_ticker


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python search.py <company name or partial ticker>")
    query = " ".join(sys.argv[1:])
    results = search_ticker(query)
    if not results:
        print(f"No matches for '{query}'.")
        return
    print(f"Matches for '{query}':")
    for r in results:
        print(f"  {r['symbol']:<8} {r['name']:<40} {r['exchange']:<8} {r['type']}")


if __name__ == "__main__":
    main()
