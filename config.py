"""
Central configuration for the stock ranker.
Tweak these values to change horizons, training universe, or scoring behavior.
"""
from pathlib import Path

# Prediction horizons in trading days
HORIZONS = [5, 10, 15, 30, 50, 200]

# How much history to pull for feature calculation + training (in days, calendar)
TRAINING_LOOKBACK_DAYS = 5 * 365   # ~5 years
LIVE_LOOKBACK_DAYS = 400           # enough history to compute all features for live rows

# Default universe used to TRAIN the models (broad + liquid = better generalization).
# You can pass --universe to override this with your own list at train time too.
DEFAULT_TRAIN_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "PG", "MA", "HD", "AVGO", "MRK", "COST", "ABBV",
    "PEP", "KO", "ADBE", "WMT", "CRM", "BAC", "MCD", "NFLX", "AMD", "TMO",
    "LIN", "CSCO", "ACN", "ABT", "DHR", "WFC", "ORCL", "TXN", "PM", "NEE",
    "DIS", "INTC", "VZ", "CMCSA", "IBM", "NKE", "UPS", "RTX", "HON", "QCOM",
]

# User-curated watchlist, grouped by theme. Sourced from the user's own stock
# screenshots/infographics (screened against yfinance so every symbol resolves
# to real price data; a few needed the correct listing, e.g. ABB Ltd -> ABBNY,
# DroneShield -> DRO.AX, Siemens -> SIEGY, Aixtron -> AIXA.DE, BE Semiconductor
# -> BESI.AS, IQE plc -> IQE.L, LPKF Laser -> LPK.DE, MKS Instruments -> MKSI,
# Corbus Pharmaceuticals -> CRBP).
WATCHLIST_SECTIONS = {
    "Mega-Cap Tech & Platforms": [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
        "AVGO", "ORCL", "IBM", "NFLX", "BABA",
    ],
    "Semiconductors & Chip Supply Chain": [
        "AMD", "INTC", "TSM", "QCOM", "MU", "ARM", "ASML", "MRVL", "ON",
        "MPWR", "NVTS", "AMKR", "STX", "WDC", "SNDK", "COHR", "LITE",
        "VIAV", "CRDO", "ALAB", "AEHR", "AAOI", "AXTI", "PENG", "SKHY",
        "AMAT", "AMS", "ASM", "CAMT", "FN", "FORM", "GLW", "IPGP", "LASR",
        "LPTH", "MTSI", "ONTO", "TER", "AIXA.DE", "BESI.AS", "IQE.L",
        "MKSI", "LPK.DE",
    ],
    "AI Infrastructure & Data Centers": [
        "VRT", "NBIS", "IREN", "CRWV", "APLD", "DOCN", "CIFR", "SMCI",
        "NET", "ETN", "NVT", "EOSE", "FLNC", "STEM", "ENVX", "MOD", "AAON",
        "DELL", "HUT", "MVST", "QS", "SLDP",
    ],
    "Cloud, SaaS & Enterprise Software": [
        "CRM", "NOW", "SNOW", "MDB", "DDOG", "TEAM", "DT", "FROG", "ESTC",
        "GTLB", "PD", "FFIV", "AKAM", "FSLY", "NTCT", "RDWR", "PANW",
        "CRWD", "FTNT", "OKTA", "ZS", "RBRK", "CHKP", "SAP", "ADP",
        "INTU", "WDAY", "SHOP", "ADBE", "APP", "ADSK", "TWLO", "U",
        "NTAP", "PATH", "PLTR", "SOUN",
    ],
    "Space, Robotics & Defense": [
        "RKLB", "ASTS", "LUNR", "RDW", "UMAC", "AVAV", "ONDS", "KTOS",
        "RTX", "GD", "LMT", "NOC", "BWXT", "DRO.AX", "SPCX", "JOBY", "PL",
        "IRDM", "MNTS", "OUST", "RCAT", "SPCE",
    ],
    "Nuclear, Power & Clean Energy": [
        "CEG", "VST", "OKLO", "SMR", "NNE", "BE", "GEV", "SU", "XYL",
        "ECL", "AWK", "PWR", "ROK", "HON", "SIEGY", "NEE", "CCJ",
        "APH", "CARR", "CAT", "CMI", "DUK", "EME", "ETR", "FIX", "FLR",
        "GNRC", "JCI", "KBR", "MTZ",
    ],
    "Quantum Computing": ["IONQ", "RGTI", "QUBT", "INFQ", "QBTS"],
    "Fintech, Payments & Consumer Platforms": [
        "SOFI", "UPST", "HOOD", "PYPL", "V", "MA", "CPNG", "UBER",
        "XPEV", "CRCL", "OSCR", "COIN", "MARA", "RIOT",
    ],
    "Healthcare & Biotech": [
        "LLY", "ABBV", "JNJ", "NVO", "ISRG", "ADPT",
        "BEAM", "CRSP", "MRNA", "RXRX", "CRBP",
    ],
    "Industrials, Financials & Diversified": [
        "ANET", "SPGI", "WM", "MPLX", "JPM", "BAC", "MELI", "SYM",
        "ZETA", "TE", "NOK", "WTW", "LOAR", "LAD", "ABG", "CMC", "WWW",
        "TXNM", "CVX", "FTAI", "SGI", "BRK-B", "COST", "ABBNY", "UHAL-B",
        "CLF",
    ],
    "Congressional Trade-Tracker Picks": [
        "ADPT", "TXNM", "WWW", "CMC", "OLED", "OKLO", "JOBY",
    ],
    "Dividend & Income ETFs": [
        "QQQI", "JEPI", "JEPQ", "GPIQ", "FEPI", "BTCI", "XYLD", "SCHD",
        "DGRO", "SDY", "VYM", "DGRW", "VYMI",
    ],
    "Broad Market & Core ETFs": [
        "QQQ", "SPY", "VOO", "IVV", "VTI", "ITOT", "SCHB", "SPTM", "VT",
        "ACWI", "SPGM", "VXUS", "IXUS", "SCHF", "SCHX", "SCHG", "SCHV",
        "VUG", "VTV", "IVW", "IVE", "SPYG", "SPYV", "BND", "AGG", "SCHZ",
        "MGC", "MGK", "VO", "VONE", "VONG", "VOOG",
    ],
    "Sector & Thematic ETFs": [
        "AIQ", "ARKX", "BLOK", "DRAM", "FBT", "IHI", "PJP", "PKB", "PPA",
        "SOXX", "VCR", "VDE", "VGT",
    ],
}

# Broader universe actually used to TRAIN the models (train.py's default).
# Separate from DEFAULT_TRAIN_UNIVERSE -- that one stays a fixed "50 popular
# stocks" list because the Ranker UI displays it and promises "50" to the
# user. This one unions it with every ticker in WATCHLIST_SECTIONS, so the
# trained models actually see examples of the smaller/more volatile names
# common in a real user's watchlist (quantum computing, space/robotics,
# etc.) instead of only mega-caps -- without that, a model asked to predict
# a stock far outside its training distribution can land in a leaf fit on
# very few, extreme historical examples and output an implausible forecast.
TRAINING_UNIVERSE = sorted(set(DEFAULT_TRAIN_UNIVERSE) | {
    t for tickers in WATCHLIST_SECTIONS.values() for t in tickers
})

# Composite score weights (must sum to 1.0 across the three signals per horizon)
SCORE_WEIGHTS = {
    "predicted_return": 0.40,   # regression model: forward return %
    "prob_up": 0.30,            # classification model: probability price is up
    "risk_adjusted": 0.30,      # predicted_return / recent volatility (Sharpe-like)
}

# LightGBM hyperparameters shared by train.py and backtest.py (both regressor
# and classifier per horizon use the same settings). Tune via tune.py, which
# scores candidates by walk-forward IC rather than a single holdout split;
# update this dict with whatever it finds best, then re-run train.py.
LGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # Require more training rows per leaf now that the universe spans mega-caps
    # down to thinly-traded small-caps -- guards against a leaf being fit on a
    # handful of extreme examples and producing an implausible prediction for
    # any ticker that lands in it. LightGBM's own default is 20.
    "min_child_samples": 30,
}

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Where backtest.py saves its walk-forward reliability summary, so other
# entry points (predict.py, analyze.py, the Streamlit app) can show which
# horizon to trust most without re-running the (multi-minute) backtest.
BACKTEST_RESULTS_PATH = MODELS_DIR / "backtest_results.json"

RANDOM_STATE = 42

# Where nightly_briefing.py writes its output. Committed to the repo (not
# gitignored) so the deployed Streamlit app can read it as a static file --
# there's no running server/database behind the deployed app, so "last
# night's data" has to live in the repo itself for the app to see it.
BRIEFING_DIR = Path(__file__).parent / "briefing"
BRIEFING_DIR.mkdir(exist_ok=True)

# Thresholds nightly_briefing.py uses to decide what counts as "notable"
# enough to call out, vs. routine day-to-day noise.
BRIEFING_FORECAST_MOVE_PP = 3.0     # predicted-return change vs. last run, in percentage points
BRIEFING_PROB_UP_MOVE_PP = 8.0      # chance-of-gain change vs. last run, in percentage points
BRIEFING_EARNINGS_WITHIN_DAYS = 7   # flag earnings reports due this soon
BRIEFING_NEAR_ATH_PCT = 2.0         # "near its all-time high" = within this % of it
BRIEFING_SENTIMENT_EXTREME = 0.35   # |mean headline sentiment score| above this is "notable"

# Forecast accuracy tracker (nightly_briefing.py): logs EVERY ticker's
# prediction for every horizon (not just ones flagged as a notable overnight
# move -- that would bias the sample toward "things already in motion"), and
# checks back once each one's horizon has elapsed to see whether it was
# actually right. A given (ticker, horizon) only gets re-logged once its
# previous entry resolves, so "pending" stays naturally capped at roughly
# one per ticker per horizon rather than growing every night.
# Caps the resolved-history list so the committed JSON file doesn't grow
# forever -- this is a rolling recent track record (sized for a few months
# at full-coverage volume), not a permanent archive (backtest.py already
# covers rigorous large-sample historical evaluation).
FORECAST_LOG_MAX_RESOLVED = 10000
# Stop waiting on a pending prediction after this many calendar days past
# its expected resolution date (data gaps, a ticker dropping out of the
# watchlist, etc. shouldn't let a pending entry linger forever).
FORECAST_LOG_STALE_AFTER_DAYS = 30
