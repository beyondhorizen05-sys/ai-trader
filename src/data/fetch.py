"""Download historical OHLCV data and persist to data/raw/."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_settings(path: Path = CONFIG_PATH) -> dict:
    """Load the YAML config file into a dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance >= 0.2.51 returns MultiIndex (field, symbol) even for one ticker.
    Collapse it down to a single level of lowercase field names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df


def fetch_symbol(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV for a single symbol. Returns a DataFrame with a DatetimeIndex."""
    logger.info("Fetching %s [%s -> %s] @ %s", symbol, start, end, interval)
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    df = _flatten_columns(df)
    df.index.name = "date"
    return df


def save_raw(df: pd.DataFrame, symbol: str, out_dir: Path = RAW_DIR) -> Path:
    """Persist a symbol's DataFrame as parquet under data/raw/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol.lower()}.parquet"
    df.to_parquet(out_path)
    logger.info("Saved %s rows to %s", len(df), out_path)
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_settings()["data"]
    for symbol in cfg["symbols"]:
        df = fetch_symbol(symbol, cfg["start"], cfg["end"], cfg["timeframe"])
        save_raw(df, symbol)


if __name__ == "__main__":
    main()