"""Symbol universe management. Provides curated lists and survivorship-bias warnings."""
from __future__ import annotations

# --- Curated universes --------------------------------------------------
# Hand-picked lists, NOT point-in-time constituents. Survivorship bias is
# present. Research acceptable, production not.

US_LARGE_CAP_30 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "CVX",
    "LLY", "ABBV", "MRK", "KO", "PEP", "AVGO", "COST", "ADBE", "CSCO",
    "TMO", "MCD", "ACN",
]

# ~100 names: the S&P 100 core plus a sprinkle of large-cap tech and mid-caps
# for cross-sectional dispersion. Hand-assembled; covers 11 GICS sectors.
US_LARGE_CAP_100 = [
    # Tech
    "AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CSCO", "CRM", "ORCL", "AMD",
    "INTC", "QCOM", "TXN", "IBM", "INTU", "NOW", "AMAT", "MU", "LRCX",
    "ADI", "KLAC",
    # Comm
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "EA",
    # Consumer Disc
    "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG",
    "GM", "F", "MAR",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL", "MDLZ", "TGT",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "BLK",
    "AXP", "SPGI", "SCHW", "CB", "MMC", "PGR",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR",
    "BMY", "AMGN", "GILD", "CVS", "MDT", "ISRG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY",
    # Industrials
    "ACN", "HON", "UPS", "BA", "CAT", "GE", "LMT", "RTX", "DE", "MMM",
    # Materials
    "LIN", "SHW", "APD", "FCX",
    # Utilities
    "NEE", "DUK", "SO", "D",
    # Real Estate
    "AMT", "PLD", "CCI", "SPG",
]

US_SECTORS_ETF = [
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
]

BROAD_ETF = ["SPY", "QQQ", "IWM", "DIA", "VTI"]


def get_universe(name: str) -> list[str]:
    universes = {
        "large_cap_30": US_LARGE_CAP_30,
        "large_cap_100": US_LARGE_CAP_100,
        "sectors": US_SECTORS_ETF,
        "broad": BROAD_ETF,
    }
    if name not in universes:
        raise KeyError(f"unknown universe {name!r}; known: {sorted(universes)}")
    return list(universes[name])


SURVIVORSHIP_WARNING = (
    "WARNING: these universes are hand-curated and subject to survivorship "
    "bias. Results on them are optimistic relative to a point-in-time "
    "universe. Use for research, not for performance claims."
)