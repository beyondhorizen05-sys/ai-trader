"""Sector maps for the curated universes. Hand-assigned GICS-ish classifications."""

SECTORS_LARGE_CAP_30 = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AVGO": "tech",
    "ADBE": "tech", "CSCO": "tech", "GOOGL": "comm", "META": "comm",
    "AMZN": "consumer_disc", "TSLA": "consumer_disc", "HD": "consumer_disc",
    "MCD": "consumer_disc", "WMT": "consumer_stap", "PG": "consumer_stap",
    "KO": "consumer_stap", "PEP": "consumer_stap", "COST": "consumer_stap",
    "BRK-B": "financials", "JPM": "financials", "V": "financials", "MA": "financials",
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "TMO": "healthcare",
    "XOM": "energy", "CVX": "energy", "ACN": "industrials",
}

SECTORS_LARGE_CAP_100 = {
    # tech
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "AVGO": "tech",
    "ADBE": "tech", "CSCO": "tech", "CRM": "tech", "ORCL": "tech",
    "AMD": "tech", "INTC": "tech", "QCOM": "tech", "TXN": "tech",
    "IBM": "tech", "INTU": "tech", "NOW": "tech", "AMAT": "tech",
    "MU": "tech", "LRCX": "tech", "ADI": "tech", "KLAC": "tech",
    # comm
    "GOOGL": "comm", "META": "comm", "NFLX": "comm", "DIS": "comm",
    "CMCSA": "comm", "TMUS": "comm", "VZ": "comm", "T": "comm", "EA": "comm",
    # consumer discretionary
    "AMZN": "consumer_disc", "TSLA": "consumer_disc", "HD": "consumer_disc",
    "MCD": "consumer_disc", "NKE": "consumer_disc", "SBUX": "consumer_disc",
    "LOW": "consumer_disc", "TJX": "consumer_disc", "BKNG": "consumer_disc",
    "GM": "consumer_disc", "F": "consumer_disc", "MAR": "consumer_disc",
    # consumer staples
    "WMT": "consumer_stap", "PG": "consumer_stap", "KO": "consumer_stap",
    "PEP": "consumer_stap", "COST": "consumer_stap", "PM": "consumer_stap",
    "MO": "consumer_stap", "CL": "consumer_stap", "MDLZ": "consumer_stap",
    "TGT": "consumer_stap",
    # financials
    "BRK-B": "financials", "JPM": "financials", "V": "financials",
    "MA": "financials", "BAC": "financials", "WFC": "financials",
    "GS": "financials", "MS": "financials", "C": "financials",
    "BLK": "financials", "AXP": "financials", "SPGI": "financials",
    "SCHW": "financials", "CB": "financials", "MMC": "financials",
    "PGR": "financials",
    # healthcare
    "UNH": "healthcare", "JNJ": "healthcare", "LLY": "healthcare",
    "ABBV": "healthcare", "MRK": "healthcare", "TMO": "healthcare",
    "ABT": "healthcare", "PFE": "healthcare", "DHR": "healthcare",
    "BMY": "healthcare", "AMGN": "healthcare", "GILD": "healthcare",
    "CVS": "healthcare", "MDT": "healthcare", "ISRG": "healthcare",
    # energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "SLB": "energy",
    "EOG": "energy", "PSX": "energy", "MPC": "energy", "OXY": "energy",
    # industrials
    "ACN": "industrials", "HON": "industrials", "UPS": "industrials",
    "BA": "industrials", "CAT": "industrials", "GE": "industrials",
    "LMT": "industrials", "RTX": "industrials", "DE": "industrials",
    "MMM": "industrials",
    # materials
    "LIN": "materials", "SHW": "materials", "APD": "materials", "FCX": "materials",
    # utilities
    "NEE": "utilities", "DUK": "utilities", "SO": "utilities", "D": "utilities",
    # real estate
    "AMT": "realestate", "PLD": "realestate", "CCI": "realestate", "SPG": "realestate",
}


def get_sectors(universe_name: str = "large_cap_30") -> dict[str, str]:
    if universe_name == "large_cap_30":
        return dict(SECTORS_LARGE_CAP_30)
    if universe_name == "large_cap_100":
        return dict(SECTORS_LARGE_CAP_100)
    raise KeyError(f"no sector map for universe {universe_name!r}")