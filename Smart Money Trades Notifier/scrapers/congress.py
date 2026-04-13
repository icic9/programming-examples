"""
U.S. House Member Stock Trade Scraper
======================================
Source: disclosures-clerk.house.gov  (official STOCK-Act disclosures)

Workflow
--------
1. Download the annual financial-disclosure ZIP → extract the XML index.
2. Find all PTR (Periodic Transaction Report) filings for the target member
   filed within the configured look-back window.
3. For each PTR PDF:
     a. Download from the clerk's server.
     b. Parse with pdfplumber to extract the transactions table.
4. Aggregate: sum purchase amounts per ticker, subtract sale amounts.
5. Return the top-N tickers ranked by net purchased value.

Transaction amounts are disclosed as dollar ranges; we use midpoint estimates.
"""

import io
import logging
import re
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pdfplumber
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://disclosures-clerk.house.gov"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Dollar-range midpoints (STOCK Act disclosure brackets) ───────────────────
_AMOUNT_MIDPOINTS = {
    # string fragment → midpoint in USD
    "1,000,001 - $5,000,000": 3_000_000,
    "500,001 - $1,000,000":   750_000,
    "250,001 - $500,000":     375_000,
    "100,001 - $250,000":     175_000,
    "50,001 - $100,000":       75_000,
    "15,001 - $50,000":        32_500,
    "1,001 - $15,000":          8_000,
    "over $5,000,000":       7_500_000,
    "over $1,000,000":       1_500_000,
    "under $1,001":               500,
}

def _parse_amount(text: str) -> float:
    """Return an approximate dollar value for a disclosed amount string."""
    for fragment, mid in _AMOUNT_MIDPOINTS.items():
        if fragment in text:
            return float(mid)
    # Generic: grab first dollar number
    nums = re.findall(r"\$?([\d,]+)", text)
    if nums:
        return float(nums[0].replace(",", ""))
    return 0.0


# ── PDF parsing ───────────────────────────────────────────────────────────────
#
# Observed House PTR PDF format (actual filing from 2026):
#
#   SP Alphabet Inc. - Class A Common P 01/16/2026 01/16/2026 $500,001 -
#   Stock (GOOGL) [ST] $1,000,000
#
# The amount range SPANS TWO LINES: the first line ends with "$500,001 -"
# and the second line starts with the upper bound "$1,000,000".
# The ticker "(GOOGL)" appears on the second line.
#
# Strategy: join any line that ends with "$ ... -" with the next line.
# This collapses the split range into one searchable string.
#
_TICKER_RE     = re.compile(r"\(([A-Z]{1,5})\)")
# Transaction type: P (Purchase), S (partial), S (full), E (Exchange)
_TYPE_RE       = re.compile(
    r"\b(P)\b"                      # Purchase
    r"|\bS\s*\(partial\)"           # Sale partial
    r"|\bS\s*\(full\)"              # Sale full
    r"|\bS\b"                       # Sale (generic)
    r"|\bE\b",                      # Exchange
    re.I,
)
# Amount range: "$1,000,001 - $5,000,000"
_RANGE_RE      = re.compile(r"\$([\d,]+)\s*[-–]\s*\$([\d,]+)")
# Fallback: any dollar amount
_DOLLAR_RE     = re.compile(r"\$(\d[\d,]*)")


def _extract_transactions_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Parse a House PTR PDF and return a list of:
        {'ticker': str, 'type': str, 'amount': float, 'raw_amount': str}
    """
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                raw_text = page.extract_text() or ""
                lines    = [l for l in raw_text.splitlines() if l.strip()]
                blocks   = _join_split_amount_lines(lines)
                for block in blocks:
                    txn = _parse_block(block)
                    if txn:
                        transactions.append(txn)
    except Exception as exc:
        logger.warning("PDF parse error: %s", exc)
    return transactions


def _join_split_amount_lines(lines: list[str]) -> list[str]:
    """
    When an amount range is split across two lines (line ends with `$X -`),
    join the two lines into one block for easier regex matching.
    """
    # Ends with a dollar amount and a dash → range continues on next line
    split_end = re.compile(r"\$[\d,]+\s*-\s*$")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if split_end.search(line) and i + 1 < len(lines):
            result.append(line + " " + lines[i + 1].strip())
            i += 2
        else:
            result.append(line)
            i += 1
    return result


def _extract_dollar_amounts(text: str) -> list[int]:
    """
    Return all dollar amounts (preceded by '$') found in text, as integers.
    Ignores anything below $100 to avoid matching prices like $15.00.
    """
    amounts = []
    for m in _DOLLAR_RE.finditer(text):
        try:
            val = int(m.group(1).replace(",", ""))
            if val >= 100:
                amounts.append(val)
        except ValueError:
            pass
    return sorted(set(amounts))


def _parse_block(text: str) -> dict | None:
    """
    Extract a single transaction from a (possibly joined) text block.
    Returns None if no valid ticker or transaction type is found.
    """
    ticker_m = _TICKER_RE.search(text)
    if not ticker_m:
        return None

    ticker  = ticker_m.group(1).upper()
    type_m  = _TYPE_RE.search(text)
    if not type_m:
        return None

    # Determine buy vs sell
    matched = type_m.group(0).strip().upper()
    if matched.startswith("P"):
        txn_type = "purchase"
    elif matched.startswith("E"):
        txn_type = "exchange"
    else:
        txn_type = "sale"

    # Amount: try direct range regex first (works when amounts are adjacent)
    range_m = _RANGE_RE.search(text)
    if range_m:
        lo  = int(range_m.group(1).replace(",", ""))
        hi  = int(range_m.group(2).replace(",", ""))
        amount  = (lo + hi) / 2
        raw_amt = range_m.group(0)
    else:
        # Fallback: extract all '$X,XXX' amounts and use the two largest
        # as the low/high bounds (avoids misreading dates as amounts)
        vals = _extract_dollar_amounts(text)
        if len(vals) >= 2:
            lo, hi = vals[-2], vals[-1]
            amount = (lo + hi) / 2
            raw_amt = f"${lo:,} - ${hi:,}"
        elif vals:
            amount  = float(vals[0])
            raw_amt = f"${vals[0]:,}"
        else:
            amount  = _parse_amount(text)
            raw_amt = ""

    return {"ticker": ticker, "type": txn_type, "amount": amount, "raw_amount": raw_amt}


# ── Index XML parsing ─────────────────────────────────────────────────────────
import xml.etree.ElementTree as ET


def _get_ptr_filings(year: int, member_last: str, member_first: str, since: datetime) -> list[tuple]:
    """
    Download the annual index ZIP and return a list of
        (doc_id, filing_date)
    for PTR filings by the target member filed on or after `since`.
    """
    zip_url = f"{BASE_URL}/public_disc/financial-pdfs/{year}FD.zip"
    logger.info("Downloading index: %s", zip_url)

    try:
        r = requests.get(zip_url, headers=_HEADERS, timeout=60)
        r.raise_for_status()
    except Exception as exc:
        logger.error("Cannot download House disclosures index for %d: %s", year, exc)
        return []

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_name = f"{year}FD.xml"
        if xml_name not in zf.namelist():
            xml_name = zf.namelist()[0]
        xml_bytes = zf.read(xml_name)

    root = ET.fromstring(xml_bytes)
    filings = []
    first_lower = member_first.lower()
    last_lower  = member_last.lower()

    for member in root.findall("Member"):
        try:
            if member.find("FilingType").text != "P":   # P = PTR
                continue
            m_last  = (member.find("Last").text  or "").lower()
            m_first = (member.find("First").text or "").lower()
            # Flexible match: last name must match; first name prefix is enough
            if m_last != last_lower:
                continue
            if not m_first.startswith(first_lower[:4]):
                continue
            doc_id      = member.find("DocID").text
            filing_date = datetime.strptime(member.find("FilingDate").text, "%m/%d/%Y")
            if filing_date >= since:
                filings.append((doc_id, filing_date))
        except Exception:
            continue

    filings.sort(key=lambda x: x[1], reverse=True)
    logger.info("Found %d PTR filings for %s %s in %d", len(filings), member_first, member_last, year)
    return filings


def _download_pdf(doc_id: str, year: int) -> bytes | None:
    url = f"{BASE_URL}/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as exc:
        logger.warning("Cannot fetch PDF %s: %s", url, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def get_congress_holdings(
    full_name: str,
    top_n: int = 10,
    days_back: int = 180,
) -> tuple[list[dict], str | None]:
    """
    Return (holdings_list, date_str) for a given House member.

    holdings_list: list of dicts with keys
        ticker, company (empty), weight (None), market_value (net USD estimate), fund (member name)
    date_str: most-recent filing date as 'YYYY-MM-DD', or None
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        logger.error("Provide a full name like 'Nancy Pelosi', got: %s", full_name)
        return [], None

    first_name, last_name = parts[0], parts[-1]
    since      = datetime.now() - timedelta(days=days_back)
    current_yr = datetime.now().year

    # Collect filings from current and previous year.
    # Always check both so the look-back window is fully covered
    # (e.g. running in January 2026 with days_back=365 needs 2025 filings too).
    all_filings: list[tuple] = []
    for yr in [current_yr, current_yr - 1]:
        all_filings += _get_ptr_filings(yr, last_name, first_name, since)

    if not all_filings:
        logger.warning("No recent PTR filings found for %s", full_name)
        return [], None

    most_recent_date = all_filings[0][1].strftime("%Y-%m-%d")

    # Aggregate transactions across all filings
    net: dict[str, float] = {}   # ticker → net USD (purchases − sales)
    processed = 0

    for doc_id, filing_date in all_filings:
        yr   = filing_date.year
        pdf  = _download_pdf(doc_id, yr)
        if not pdf:
            continue
        txns = _extract_transactions_from_pdf(pdf)
        for t in txns:
            ticker = t["ticker"]
            sign   = -1.0 if "sale" in t["type"] else 1.0
            net[ticker] = net.get(ticker, 0.0) + sign * t["amount"]
        processed += 1
        time.sleep(0.5)   # be polite to the server

    if not net:
        logger.warning("Parsed %d PDF(s) for %s but extracted 0 transactions", processed, full_name)
        return [], most_recent_date

    # Sort by net value (largest net purchase first); exclude net-sold positions
    ranked = sorted(
        [(ticker, val) for ticker, val in net.items() if val > 0],
        key=lambda x: x[1],
        reverse=True,
    )

    holdings = [
        {
            "ticker":       ticker,
            "company":      "",
            "weight":       None,
            "market_value": val,
            "fund":         full_name,
        }
        for ticker, val in ranked[:top_n]
    ]

    logger.info(
        "%s: %d tickers with net purchases from %d filing(s)",
        full_name, len(holdings), processed,
    )
    return holdings, most_recent_date


# ── Asset-type detection ──────────────────────────────────────────────────────

_OPTION_TYPE_RE = re.compile(r"\[(OP)\]", re.I)        # [OP] = option
_CALL_RE        = re.compile(r"\bcall\b",  re.I)
_PUT_RE         = re.compile(r"\bput\b",   re.I)

def _detect_instrument(block: str) -> str:
    """Return 'Stock', 'Call Option', 'Put Option', or 'Option' for a block."""
    if _OPTION_TYPE_RE.search(block):
        if _CALL_RE.search(block):
            return "Call Option"
        if _PUT_RE.search(block):
            return "Put Option"
        return "Option"
    return "Stock"


def _action_label(txn_type: str, instrument: str) -> str:
    """Human-readable action label combining transaction type and instrument."""
    if txn_type == "purchase":
        return f"Bought {instrument}"
    if txn_type == "exchange":
        return f"Exchange / Spinoff"
    return f"Sold {instrument}"


def get_congress_trades(
    full_name: str,
    days_back: int = 180,
) -> tuple[list[dict], str | None]:
    """
    Return (trades_list, most_recent_date) — all individual disclosed transactions
    for a House member within the look-back window.

    Each trade dict has:
        member, filing_date, ticker, instrument, action, amount_lo, amount_hi,
        amount_mid, raw_amount, is_long (True for stocks/calls, False for puts)
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        return [], None

    first_name, last_name = parts[0], parts[-1]
    since      = datetime.now() - timedelta(days=days_back)
    current_yr = datetime.now().year

    all_filings: list[tuple] = []
    for yr in [current_yr, current_yr - 1]:
        all_filings += _get_ptr_filings(yr, last_name, first_name, since)

    if not all_filings:
        return [], None

    most_recent_date = all_filings[0][1].strftime("%Y-%m-%d")
    trades: list[dict] = []

    for doc_id, filing_date in all_filings:
        yr  = filing_date.year
        pdf = _download_pdf(doc_id, yr)
        if not pdf:
            continue

        # Re-run the block-level parsing so we have per-block context
        try:
            with pdfplumber.open(io.BytesIO(pdf)) as pdf_obj:
                for page in pdf_obj.pages:
                    raw = page.extract_text() or ""
                    lines  = [l for l in raw.splitlines() if l.strip()]
                    blocks = _join_split_amount_lines(lines)
                    for block in blocks:
                        txn = _parse_block(block)
                        if not txn:
                            continue
                        instrument = _detect_instrument(block)
                        is_long    = txn["type"] != "sale" or instrument == "Put Option"
                        # Put options are bearish (short-equivalent) even when "purchased"
                        if instrument == "Put Option" and txn["type"] == "purchase":
                            is_long = False
                        trades.append({
                            "member":       full_name,
                            "filing_date":  filing_date.strftime("%Y-%m-%d"),
                            "ticker":       txn["ticker"],
                            "instrument":   instrument,
                            "action":       _action_label(txn["type"], instrument),
                            "amount_mid":   txn["amount"],
                            "raw_amount":   txn["raw_amount"],
                            "is_long":      is_long,
                        })
        except Exception as exc:
            logger.warning("Trade-level parse error for %s doc %s: %s", full_name, doc_id, exc)
        time.sleep(0.5)

    trades.sort(key=lambda x: x["filing_date"], reverse=True)
    logger.info("%s: %d individual trades from %d filing(s)", full_name, len(trades), len(all_filings))
    return trades, most_recent_date
