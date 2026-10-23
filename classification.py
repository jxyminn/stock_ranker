"""
Auto-classifies a new ticker into the best-matching section of
config.WATCHLIST_SECTIONS, so a user adding a stock from the app doesn't
need anyone to manually decide where it belongs.

The industry/sector -> section tables below were derived empirically: for
every ticker already in WATCHLIST_SECTIONS, we pulled its real yfinance
sector/industry and took the majority-vote section per industry/sector
(excluding "Congressional Trade-Tracker Picks" and "Mega-Cap Tech &
Platforms" from that vote, since those two are thematic/size-based rather
than industry-based -- a new ticker should never be auto-sorted into "stocks
a politician happened to trade", and mega-cap status is handled separately
below via a market-cap check instead).

This is inherently approximate -- these sections are curated/thematic, not
a formal industry taxonomy, so some new tickers will land in a plausible-but-
imperfect bucket. Anything that doesn't match well goes to NEW_ADDITIONS_SECTION
instead of being forced into the wrong place.
"""
NEW_ADDITIONS_SECTION = "New Additions"

MEGA_CAP_MARKET_CAP = 300_000_000_000  # $300B+
MEGA_CAP_SECTORS = {"Technology", "Communication Services", "Consumer Cyclical"}

INDUSTRY_TO_SECTION = {
    "Advertising Agencies": "Cloud, SaaS & Enterprise Software",
    "Aerospace & Defense": "Space, Robotics & Defense",
    "Auto & Truck Dealerships": "Industrials, Financials & Diversified",
    "Auto Manufacturers": "Fintech, Payments & Consumer Platforms",
    "Auto Parts": "AI Infrastructure & Data Centers",
    "Banks - Diversified": "Industrials, Financials & Diversified",
    "Building Products & Equipment": "AI Infrastructure & Data Centers",
    "Capital Markets": "Fintech, Payments & Consumer Platforms",
    "Communication Equipment": "Semiconductors & Chip Supply Chain",
    "Computer Hardware": "Semiconductors & Chip Supply Chain",
    "Conglomerates": "Nuclear, Power & Clean Energy",
    "Credit Services": "Fintech, Payments & Consumer Platforms",
    "Discount Stores": "Industrials, Financials & Diversified",
    "Drug Manufacturers - General": "Healthcare & Biotech",
    "Electrical Equipment & Parts": "AI Infrastructure & Data Centers",
    "Engineering & Construction": "Nuclear, Power & Clean Energy",
    "Financial Data & Stock Exchanges": "Industrials, Financials & Diversified",
    "Furnishings, Fixtures & Appliances": "Industrials, Financials & Diversified",
    "Healthcare Plans": "Fintech, Payments & Consumer Platforms",
    "Information Technology Services": "AI Infrastructure & Data Centers",
    "Insurance - Diversified": "Industrials, Financials & Diversified",
    "Insurance Brokers": "Industrials, Financials & Diversified",
    "Internet Content & Information": "AI Infrastructure & Data Centers",
    "Internet Retail": "Fintech, Payments & Consumer Platforms",
    "Medical Instruments & Supplies": "Healthcare & Biotech",
    "Oil & Gas Integrated": "Nuclear, Power & Clean Energy",
    "Oil & Gas Midstream": "Industrials, Financials & Diversified",
    "Rental & Leasing Services": "Industrials, Financials & Diversified",
    "Scientific & Technical Instruments": "Semiconductors & Chip Supply Chain",
    "Semiconductor Equipment & Materials": "Semiconductors & Chip Supply Chain",
    "Semiconductors": "Semiconductors & Chip Supply Chain",
    "Software - Application": "Cloud, SaaS & Enterprise Software",
    "Software - Infrastructure": "Cloud, SaaS & Enterprise Software",
    "Specialty Chemicals": "Nuclear, Power & Clean Energy",
    "Specialty Industrial Machinery": "Nuclear, Power & Clean Energy",
    "Uranium": "Nuclear, Power & Clean Energy",
    "Utilities - Independent Power Producers": "Nuclear, Power & Clean Energy",
    "Utilities - Regulated Electric": "Nuclear, Power & Clean Energy",
    "Utilities - Regulated Water": "Nuclear, Power & Clean Energy",
    "Utilities - Renewable": "AI Infrastructure & Data Centers",
    "Waste Management": "Industrials, Financials & Diversified",
    # A few industries common among newer/speculative names that weren't
    # well-represented enough in the original 210 to earn a majority vote.
    "Software - Infrastructure ": "Cloud, SaaS & Enterprise Software",
    "Biotechnology": "Healthcare & Biotech",
    "Diagnostics & Research": "Healthcare & Biotech",
    "Airlines": "Space, Robotics & Defense",
}

SECTOR_TO_SECTION = {
    "Basic Materials": "Nuclear, Power & Clean Energy",
    "Communication Services": "AI Infrastructure & Data Centers",
    "Consumer Cyclical": "Industrials, Financials & Diversified",
    "Consumer Defensive": "Industrials, Financials & Diversified",
    "Energy": "Nuclear, Power & Clean Energy",
    "Financial Services": "Fintech, Payments & Consumer Platforms",
    "Healthcare": "Healthcare & Biotech",
    "Industrials": "Space, Robotics & Defense",
    "Real Estate": "Industrials, Financials & Diversified",
    "Technology": "Cloud, SaaS & Enterprise Software",
    "Utilities": "Nuclear, Power & Clean Energy",
}


def classify_section(fund: dict) -> str:
    """fund: a data_fetcher.fetch_fundamentals()-shaped dict. Returns the
    best-matching section name, or NEW_ADDITIONS_SECTION if nothing fits."""
    sector = fund.get("sector")
    industry = fund.get("industry")
    market_cap = fund.get("market_cap") or 0
    dividend_yield = fund.get("dividend_yield") or 0

    # No sector/industry from yfinance almost always means an ETF.
    if not sector and not industry:
        return "Dividend & Income ETFs" if dividend_yield > 0.03 else "Broad Market & Core ETFs"

    if market_cap > MEGA_CAP_MARKET_CAP and sector in MEGA_CAP_SECTORS:
        return "Mega-Cap Tech & Platforms"

    if industry in INDUSTRY_TO_SECTION:
        return INDUSTRY_TO_SECTION[industry]

    if sector in SECTOR_TO_SECTION:
        return SECTOR_TO_SECTION[sector]

    return NEW_ADDITIONS_SECTION
