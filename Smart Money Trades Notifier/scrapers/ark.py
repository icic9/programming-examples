"""
ARK Invest ETF Holdings Scraper
================================
Primary source: arkfunds.io  (unofficial free API — clean JSON with daily data)
  https://arkfunds.io/api/v2/etf/holdings?symbol=ARKK

Fallback: ark-funds.com CSV (official, if the direct URL still works)
  https://ark-funds.com/wp-content/uploads/funds-etf-csv/<filename>.csv
"""

import logging
import re
from io import StringIO

import requests
import pandas as pd

logger = logging.getLogger(__name__)

ARK_FUND_FILES = {
    "ARKK": "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
    "ARKW": "ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.csv",
    "ARKG": "ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
    "ARKQ": "ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.csv",
    "ARKF": "ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.csv",
    "ARKX": "ARK_SPACE_EXPLORATION_&_INNOVATION_ETF_ARKX_HOLDINGS.csv",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/csv, */*",
}


# ── Source 1: arkfunds.io API ─────────────────────────────────────────────────

def _from_arkfunds_io(fund: str, top_n: int):
    """
    Fetch holdings from the free arkfunds.io API.
    Returns (holdings_list, date_str) or ([], None) on failure.
    """
    url = f"https://arkfunds.io/api/v2/etf/holdings?symbol={fund}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        raw_holdings = data.get("holdings", [])
        if not raw_holdings:
            return [], None

        # Already sorted by weight_rank; take top_n
        date_str = raw_holdings[0].get("date") if raw_holdings else None
        results = []
        for h in raw_holdings[:top_n]:
            ticker = (h.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            results.append(
                {
                    "ticker":       ticker,
                    "company":      h.get("company", ""),
                    "weight":       h.get("weight"),
                    "market_value": h.get("market_value"),
                    "fund":         fund,
                }
            )
        logger.info("ARK %s: fetched %d holdings via arkfunds.io (as of %s)", fund, len(results), date_str)
        return results, date_str
    except Exception as exc:
        logger.warning("arkfunds.io fetch failed for %s: %s", fund, exc)
        return [], None


# ── Source 2: official ark-funds.com CSV ─────────────────────────────────────

def _parse_csv(text: str) -> pd.DataFrame:
    lines = [l for l in text.splitlines() if l.strip()]
    header_idx = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if "ticker" in low and ("fund" in low or "company" in low or "weight" in low):
            header_idx = i
            break
    df = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        for c in df.columns
    ]
    return df


def _from_ark_csv(fund: str, top_n: int):
    filename = ARK_FUND_FILES.get(fund, ARK_FUND_FILES["ARKK"])
    url = f"https://ark-funds.com/wp-content/uploads/funds-etf-csv/{filename}"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=30, allow_redirects=True)
        if r.status_code != 200 or "html" in r.headers.get("content-type", "").lower():
            return [], None
        df = _parse_csv(r.text)

        for col in ("ticker", "symbol"):
            if col in df.columns and col != "ticker":
                df = df.rename(columns={col: "ticker"})
        if "ticker" not in df.columns:
            return [], None

        df = df[df["ticker"].notna() & (df["ticker"].astype(str).str.strip() != "")]
        weight_col = next((c for c in df.columns if "weight" in c), None)
        if weight_col:
            df[weight_col] = pd.to_numeric(
                df[weight_col].astype(str).str.replace("%", "", regex=False).str.strip(),
                errors="coerce",
            )
            df = df.dropna(subset=[weight_col]).sort_values(weight_col, ascending=False)

        date_str   = str(df["date"].iloc[0]).strip() if "date" in df.columns and len(df) > 0 else None
        company_col = next((c for c in ("company", "name", "security") if c in df.columns), None)
        mv_col      = next((c for c in df.columns if "market" in c and "value" in c), None)

        results = []
        for _, row in df.head(top_n).iterrows():
            ticker = str(row["ticker"]).strip().upper()
            if not ticker or ticker in ("NAN", "-", ""):
                continue
            results.append(
                {
                    "ticker":       ticker,
                    "company":      str(row[company_col]).strip() if company_col else "",
                    "weight":       float(row[weight_col]) if weight_col and pd.notna(row.get(weight_col)) else None,
                    "market_value": row.get(mv_col),
                    "fund":         fund,
                }
            )
        logger.info("ARK %s: fetched %d holdings via official CSV (as of %s)", fund, len(results), date_str)
        return results, date_str
    except Exception as exc:
        logger.warning("ARK CSV fetch failed for %s: %s", fund, exc)
        return [], None


# ── Public API ────────────────────────────────────────────────────────────────

def get_ark_holdings(fund: str = "ARKK", top_n: int = 10):
    """
    Return (holdings_list, date_str) for the given ARK ETF.

    holdings_list is a list of dicts:
        ticker, company, weight (%), market_value, fund
    date_str is the holdings date as 'YYYY-MM-DD', or None.
    """
    fund = fund.upper()

    # Try arkfunds.io first (most reliable currently)
    holdings, date_str = _from_arkfunds_io(fund, top_n)
    if holdings:
        return holdings, date_str

    # Fallback: official ark-funds.com CSV
    holdings, date_str = _from_ark_csv(fund, top_n)
    if holdings:
        return holdings, date_str

    logger.error(
        "ARK %s: all sources failed. "
        "Visit https://ark-funds.com/funds/%s/ to download the CSV manually "
        "and place it as data/ark_%s.csv",
        fund, fund.lower(), fund.lower(),
    )
    return [], None
