from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"

TREASURY_ETFS = {
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "EDV": "Vanguard Extended Duration Treasury ETF",
    "GOVT": "iShares U.S. Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
}

RISK_FREE_TICKER = "^IRX"


class DataError(RuntimeError):
    pass


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.replace('^', '').lower()}.csv"


def _download_close(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start="1990-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise DataError(f"No data returned for {ticker}. Check your internet connection.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.name = ticker
    return close


def load_close(ticker: str, refresh: bool = False) -> pd.Series:
    """Dividend-adjusted daily closes, cached as CSV under ./data."""
    path = _cache_path(ticker)
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s
    s = _download_close(ticker)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(path)
    return s


def load_risk_free(refresh: bool = False) -> pd.Series:
    """Daily risk-free return from the 13-week T-bill yield (^IRX, in percent)."""
    irx = load_close(RISK_FREE_TICKER, refresh=refresh)
    return (irx / 100.0 / 252.0).rename("rf")


def load_option_inputs(ticker: str = "TLT", refresh: bool = False) -> pd.DataFrame:
    """Unadjusted closes (option strikes are on the traded price), the MOVE index and dividends."""
    path = CACHE_DIR / f"{ticker.lower()}_option_inputs.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = yf.download(ticker, start="1990-01-01", auto_adjust=False, actions=True, progress=False)
    move = yf.download("^MOVE", start="1990-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty or move is None or move.empty:
        raise DataError(f"Could not download option inputs for {ticker}.")

    def col(frame: pd.DataFrame, name: str) -> pd.Series:
        s = frame[name]
        return s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s

    df = pd.DataFrame({"close": col(raw, "Close"), "dividend": col(raw, "Dividends")})
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    m = col(move, "Close")
    m.index = pd.DatetimeIndex(m.index).tz_localize(None).normalize()
    df["move"] = m.reindex(df.index).ffill()
    df = df.dropna(subset=["close", "move"])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_dataset(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Daily asset returns and risk-free returns on the asset's trading calendar."""
    close = load_close(ticker, refresh=refresh)
    rf = load_risk_free(refresh=refresh)
    df = pd.DataFrame({"close": close})
    # The return earned on day t accrues at the rate known at the close of t-1.
    df["rf"] = rf.reindex(df.index, method="ffill").shift(1)
    df["ret"] = df["close"].pct_change()
    df = df.dropna()
    if df.empty:
        raise DataError(f"Not enough overlapping data for {ticker}.")
    return df
