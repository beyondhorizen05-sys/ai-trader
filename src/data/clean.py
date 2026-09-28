"""Validate and normalize raw OHLCV data; persist to data/processed/."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OHLCV_COLS = ["open", "high", "low", "close", "adj close", "volume"]


def load_settings(path: Path = CONFIG_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw(symbol: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load a raw parquet file for a symbol."""
    path = raw_dir / f"{symbol.lower()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No raw data for {symbol} at {path}")
    return pd.read_parquet(path)


def validate(df: pd.DataFrame, symbol: str) -> None:
    """Raise ValueError if the DataFrame fails any structural checks."""
    missing = [c for c in OHLCV_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    if df.empty:
        raise ValueError(f"{symbol}: empty DataFrame")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: index is not a DatetimeIndex")

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"{symbol}: duplicate dates: {dupes[:5]}")

    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: index is not sorted ascending")

    if df[OHLCV_COLS].isna().any().any():
        cols = df[OHLCV_COLS].isna().any()
        raise ValueError(f"{symbol}: NaNs in columns {cols[cols].index.tolist()}")

    if (df["high"] < df["low"]).any():
        raise ValueError(f"{symbol}: high < low on {int((df['high'] < df['low']).sum())} rows")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: high < max(open, close) on some rows")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol}: low > min(open, close) on some rows")

    if (df["volume"] < 0).any():
        raise ValueError(f"{symbol}: negative volume on some rows")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, dedupe, and enforce dtypes."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.copy()
    df["volume"] = df["volume"].astype("int64")
    for c in ["open", "high", "low", "close", "adj close"]:
        df[c] = df[c].astype("float64")
    df.index.name = "date"
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample daily OHLCV to a higher timeframe. Volume is summed."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "adj close": "last",
        "volume": "sum",
    }
    out = df.resample(rule).agg(agg).dropna(how="any")
    out.index.name = "date"
    return out


def save_processed(df: pd.DataFrame, symbol: str, suffix: str = "", out_dir: Path = PROCESSED_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{symbol.lower()}{('_' + suffix) if suffix else ''}.parquet"
    out_path = out_dir / name
    df.to_parquet(out_path)
    logger.info("Saved %s rows to %s", len(df), out_path)
    return out_path


def clean_symbol(symbol: str, resample_rules: list[str] | None = None) -> None:
    logger.info("Cleaning %s", symbol)
    raw = load_raw(symbol)
    daily = normalize(raw)          # sort + dedupe + dtype first
    validate(daily, symbol)         # then validate the canonical form
    save_processed(daily, symbol, suffix="1d")
    for rule in (resample_rules or []):
        rs = resample_ohlcv(daily, rule)
        save_processed(rs, symbol, suffix=rule)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_settings()["data"]
    rules = cfg.get("resample", [])  # e.g. ["1W", "1ME"]
    for symbol in cfg["symbols"]:
        clean_symbol(symbol, resample_rules=rules)


if __name__ == "__main__":
    main()