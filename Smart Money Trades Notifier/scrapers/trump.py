"""
Trump / Presidential Financial Disclosure Scraper
==================================================
Sources used in priority order:

1. SEC EDGAR Form 4  —  CIK 0000947033 (TRUMP DONALD J)
   Trump's Trust holds DJT (Trump Media) shares.  Every change is filed as
   a Form 4 (insider ownership report).  This gives us his current DJT stake
   automatically, and we price it with a live Yahoo Finance quote.

2. White House Disclosures page  —  whitehouse.gov/disclosures/
   Since Jan 2025 the President files Periodic Transaction Reports (PTRs),
   the same STOCK Act disclosures required of Congress members.  PDFs are
   listed publicly.  Because they are image-scanned (not text PDFs) we
   optionally send them to an OpenRouter LLM (vision) for extraction when
   OPENROUTER_API_KEY is set in .env.

3. Manual data file  —  data/trump_holdings.json
   Static fallback for non-DJT holdings (AAPL, AMZN, NVDA, etc.) from the
   most-recently filed OGE Form 278e annual disclosure.  These rarely change
   and the OGE PDF is image-based (not parseable without OCR).  Update this
   file whenever a new annual disclosure is published.

NOTE: Presidential disclosures are annual for full holdings (OGE Form 278e)
and periodic for new trades (PTRs).  DJT stake is tracked in real-time via
SEC Form 4; other holdings may lag by up to 12 months.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_SEC_HEADERS = {
    "User-Agent": "SmartMoneyTracker contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, application/xml, text/html, */*",
}

_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
}

_TRUMP_CIK       = "0000947033"
_DATA_FILE       = Path(__file__).parent.parent / "data" / "trump_holdings.json"
_WHITEHOUSE_DISC = "https://www.whitehouse.gov/disclosures/"

# OGE Form 278e amount category midpoints (A–L)
_OGE_AMOUNTS = {
    "A": 500,
    "B": 8_000,
    "C": 32_500,
    "D": 75_000,
    "E": 175_000,
    "F": 375_000,
    "G": 750_000,
    "H": 3_000_000,
    "I": 7_500_000,
    "J": 17_500_000,
    "K": 37_500_000,
    "L": 75_000_000,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source 1 — SEC EDGAR: Form 4 for DJT stake
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_djt_price() -> float | None:
    """Fetch the current DJT (Trump Media) price from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/DJT?interval=1d&range=1d"
        r = requests.get(url, headers=_WEB_HEADERS, timeout=10)
        r.raise_for_status()
        d = r.json()
        price = d["chart"]["result"][0]["meta"]["regularMarketPrice"]
        return float(price)
    except Exception as exc:
        logger.debug("Yahoo Finance DJT price fetch failed: %s", exc)
    return None


def _parse_form4_xml(xml_text: str) -> dict:
    """
    Parse a Form 4 XML and return:
      {
        "direct_shares":   int,   # shares owned directly
        "indirect_shares": int,   # shares owned via trust/entity
        "period":          str,   # YYYY-MM-DD
        "ticker":          str,
      }
    """
    import xml.etree.ElementTree as ET

    result = {
        "direct_shares": 0,
        "indirect_shares": 0,
        "period": None,
        "ticker": "DJT",
    }

    try:
        root = ET.fromstring(xml_text)

        def _elem_val(parent, tag: str) -> str:
            """Extract text from <tag><value>TEXT</value></tag> or <tag>TEXT</tag>."""
            el = parent.find(f".//{tag}")
            if el is None:
                return ""
            value_child = el.find("value")
            if value_child is not None and value_child.text:
                return value_child.text.strip()
            return (el.text or "").strip()

        period_str = _elem_val(root, "periodOfReport")
        if not period_str:
            # periodOfReport is a direct text element, not wrapped in <value>
            period_el = root.find(".//periodOfReport")
            if period_el is not None:
                period_str = (period_el.text or "").strip()
        if period_str:
            result["period"] = period_str

        ticker_str = _elem_val(root, "issuerTradingSymbol")
        if not ticker_str:
            ticker_el = root.find(".//issuerTradingSymbol")
            if ticker_el is not None:
                ticker_str = (ticker_el.text or "").strip()
        if ticker_str:
            result["ticker"] = ticker_str.upper()

        def _elem_val(parent, tag: str) -> str:
            """Extract text from <tag><value>TEXT</value></tag> or <tag>TEXT</tag>."""
            el = parent.find(f".//{tag}")
            if el is None:
                return ""
            # Form 4 wraps values in a <value> child
            value_child = el.find("value")
            if value_child is not None and value_child.text:
                return value_child.text.strip()
            return (el.text or "").strip()

        # --- Non-derivative holdings (holdings section, not transactions) ---
        for holding in root.findall(".//nonDerivativeHolding"):
            shares_str = _elem_val(holding, "sharesOwnedFollowingTransaction")
            if not shares_str:
                continue
            try:
                shares = int(float(shares_str))
            except (ValueError, TypeError):
                continue

            ownership = _elem_val(holding, "directOrIndirectOwnership").upper() or "D"
            if ownership == "I":
                result["indirect_shares"] = max(result["indirect_shares"], shares)
            else:
                result["direct_shares"] = max(result["direct_shares"], shares)

        # --- Non-derivative transactions: take the post-transaction value ---
        for txn in root.findall(".//nonDerivativeTransaction"):
            shares_str = _elem_val(txn, "sharesOwnedFollowingTransaction")
            if not shares_str:
                continue
            try:
                shares = int(float(shares_str))
            except (ValueError, TypeError):
                continue

            ownership = _elem_val(txn, "directOrIndirectOwnership").upper() or "D"
            if ownership == "I":
                result["indirect_shares"] = max(result["indirect_shares"], shares)
            else:
                result["direct_shares"] = max(result["direct_shares"], shares)

    except Exception as exc:
        logger.warning("Form 4 XML parse error: %s", exc)

    return result


def _from_sec_edgar_djt() -> tuple[dict | None, str | None]:
    """
    Fetch Trump's most-recent Form 4 filing from SEC EDGAR.
    Returns (holding_dict, date_str) or (None, None) on failure.

    holding_dict format:
        ticker, company, weight (None), market_value (USD), fund='Trump'
    """
    try:
        submissions_url = f"https://data.sec.gov/submissions/CIK{_TRUMP_CIK}.json"
        r = requests.get(submissions_url, headers=_SEC_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.warning("SEC EDGAR submissions fetch failed: %s", exc)
        return None, None

    # Find the most recent Form 4
    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    accns   = filings.get("accessionNumber", [])
    dates   = filings.get("filingDate", [])

    latest_form4_accn = None
    latest_form4_date = None
    for form, accn, dt in zip(forms, accns, dates):
        if form == "4":
            latest_form4_accn = accn
            latest_form4_date = dt
            break   # submissions are sorted newest-first

    if not latest_form4_accn:
        logger.info("SEC EDGAR: no Form 4 found for Trump CIK %s", _TRUMP_CIK)
        return None, None

    # Fetch the Form 4 XML
    acc_clean = latest_form4_accn.replace("-", "")
    cik_clean = _TRUMP_CIK.lstrip("0")
    xml_url   = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{acc_clean}/primary_doc.xml"

    try:
        rx = requests.get(xml_url, headers=_SEC_HEADERS, timeout=20)
        rx.raise_for_status()
        form4_data = _parse_form4_xml(rx.text)
    except Exception as exc:
        logger.warning("Form 4 XML fetch/parse failed (%s): %s", xml_url, exc)
        return None, None

    # Total shares = direct + indirect (trust)
    total_shares = form4_data["direct_shares"] + form4_data["indirect_shares"]
    if total_shares == 0:
        logger.info("SEC EDGAR Form 4: zero shares found for DJT — possibly stale filing.")
        return None, None

    # Get current price
    price = _get_djt_price()
    if price is None:
        logger.warning("Could not fetch DJT price; using last known $9.51")
        price = 9.51   # reasonable fallback

    market_value = total_shares * price
    period       = form4_data.get("period") or latest_form4_date

    logger.info(
        "SEC EDGAR: DJT stake = %s shares (direct=%s indirect=%s) @ $%.2f = $%s  [Form 4: %s]",
        f"{total_shares:,}",
        f"{form4_data['direct_shares']:,}",
        f"{form4_data['indirect_shares']:,}",
        price,
        f"{market_value:,.0f}",
        period,
    )

    holding = {
        "ticker":       form4_data["ticker"],
        "company":      "Trump Media & Technology Group Corp.",
        "weight":       None,
        "market_value": market_value,
        "fund":         "Trump",
        "notes":        f"{total_shares:,} shares via Trust @ ${price:.2f}  (Form 4 {period})",
    }
    return holding, period


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source 2 — White House Disclosures: PTR PDFs (trades)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _scrape_whitehouse_ptrs(days_back: int = 365) -> list[tuple[str, str]]:
    """
    Scrape the White House disclosures page for Trump PTR PDF links.
    Returns list of (pdf_url, date_str) pairs, newest first.
    date_str is extracted from the filename, e.g. '2.26.26' → '2026-02-26'.
    """
    try:
        r = requests.get(_WHITEHOUSE_DISC, headers=_WEB_HEADERS, timeout=20)
        r.raise_for_status()
    except Exception as exc:
        logger.warning("White House disclosures page fetch failed: %s", exc)
        return []

    # Match Trump PTR PDF links, e.g.:
    #   /wp-content/uploads/2026/03/President-Donald-J.-Trump-Periodic-Transaction-Report-2.26.26-1.pdf
    ptr_re = re.compile(
        r'href="(https?://www\.whitehouse\.gov/wp-content/uploads/\d{4}/\d{2}/'
        r'President-Donald-J\.-Trump-Periodic-Transaction-Report-[^"]+\.pdf)"',
        re.I,
    )

    cutoff = datetime.now() - timedelta(days=days_back)
    results: list[tuple[str, str]] = []

    for m in ptr_re.finditer(r.text):
        url  = m.group(1)
        # Extract date from filename, e.g. "2.26.26" → "2026-02-26"
        date_match = re.search(r'(\d{1,2})\.(\d{1,2})\.(\d{2,4})(?:-\d+)?\.pdf', url)
        if date_match:
            mo, dy, yr = date_match.groups()
            yr = int(yr)
            yr = 2000 + yr if yr < 100 else yr
            try:
                dt = datetime(yr, int(mo), int(dy))
                if dt >= cutoff:
                    results.append((url, dt.strftime("%Y-%m-%d")))
            except ValueError:
                pass

    # De-duplicate exact same URLs, keep all parts
    seen_urls: set[str] = set()
    unique: list[tuple[str, str]] = []
    for url, date_str in results:
        if url not in seen_urls:
            seen_urls.add(url)
            unique.append((url, date_str))

    # Sort newest first (by date, then by URL for stable ordering of parts)
    unique.sort(key=lambda x: (x[1], x[0]), reverse=True)

    logger.info("White House: found %d PTR PDF(s) within %d-day window", len(unique), days_back)
    return unique


def _parse_ptr_pdf_via_llm(pdf_url: str, api_key: str) -> list[dict]:
    """
    Send a White House PTR PDF to OpenRouter for extraction.
    Returns list of trade dicts:
        {filing_date, ticker, instrument, action, amount_mid, raw_amount, is_long}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    prompt = (
        "This is a US Presidential Periodic Transaction Report (PTR). "
        "Extract ALL transactions shown — stocks, bonds, ETFs, options, and any other instruments. "
        "Return ONLY a JSON array with no markdown, no extra text. "
        "Each object must have these keys:\n"
        '  "date": transaction date as YYYY-MM-DD (use notification date if transaction date missing),\n'
        '  "ticker": stock ticker symbol if available (uppercase, e.g. AAPL), else empty string,\n'
        '  "company": full description or company name (include bond name/maturity if a bond),\n'
        '  "instrument": "Stock", "Bond", "Option", "ETF", or "Other",\n'
        '  "action": "Purchase" or "Sale (Full)" or "Sale (Partial)" or "Exchange",\n'
        '  "amount_range": the dollar range as printed, e.g. "$1,001 - $15,000",\n'
        '  "option_type": "Call" or "Put" or "" (blank if not an option)\n'
        "If a field cannot be determined, use an empty string. "
        "Do NOT skip any transaction — include bonds and fixed-income instruments too."
    )

    payload = {
        "model":      os.getenv("TRUMP_PTR_MODEL", "google/gemini-2.0-flash-lite-001"),
        "max_tokens": 16000,
        "messages": [
            {
                "role":    "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "file", "file": {"filename": "ptr.pdf", "file_data": pdf_url}},
                ],
            }
        ],
        "plugins": [
            {
                "id":  "file-parser",
                "pdf": {"engine": "mistral-ocr"},
            }
        ],
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())
        try:
            raw = json.loads(content)
        except json.JSONDecodeError:
            # Truncated output: salvage complete objects already in the array
            # Find the last complete '}' before the cut-off and close the array
            last_close = content.rfind("}")
            if last_close != -1:
                truncated = content[: last_close + 1] + "\n]"
                try:
                    raw = json.loads(truncated)
                    logger.info("  (truncated JSON salvaged — %d items recovered)", len(raw))
                except json.JSONDecodeError:
                    logger.warning("LLM PTR parse failed for %s: truncated JSON unrecoverable", pdf_url)
                    return []
            else:
                logger.warning("LLM PTR parse failed for %s: no valid JSON objects found", pdf_url)
                return []
        if not isinstance(raw, list):
            logger.warning("LLM returned non-list for PTR: %r", raw)
            return []
    except Exception as exc:
        logger.warning("LLM PTR parse failed for %s: %s", pdf_url, exc)
        return []

    trades = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ticker     = (item.get("ticker") or "").strip().upper()
        company    = (item.get("company") or "").strip()
        instrument = (item.get("instrument") or "Stock").strip()

        # Must have at least a description to be useful
        if not ticker and not company:
            continue
        # Ignore malformed tickers (too long) but still keep the row; blank the ticker
        if len(ticker) > 6:
            ticker = ""

        action = (item.get("action") or "").strip()
        is_long = True
        opt_type = (item.get("option_type") or "").strip().lower()
        if "put" in opt_type:
            is_long = False

        # Parse amount range → midpoint
        raw_amt = (item.get("amount_range") or "").strip()
        nums    = re.findall(r"[\d,]+", raw_amt)
        if len(nums) >= 2:
            amount_mid = (int(nums[0].replace(",", "")) + int(nums[1].replace(",", ""))) / 2
        elif nums:
            amount_mid = int(nums[0].replace(",", ""))
        else:
            amount_mid = 0.0

        date_str = (item.get("date") or "").strip()

        trades.append({
            "filing_date": date_str,
            "ticker":      ticker,
            "company":     company,
            "instrument":  instrument,
            "action":      action,
            "amount_mid":  amount_mid,
            "raw_amount":  raw_amt,
            "is_long":     is_long,
        })

    return trades


def get_trump_trades(days_back: int = 365) -> tuple[list[dict], str | None]:
    """
    Return (trades_list, latest_date_str) from Trump's White House PTR filings.

    Requires OPENROUTER_API_KEY in .env to parse the image-based PDFs.
    Returns empty list if API key is not set.

    Each trade dict:
        filing_date, ticker, company, instrument, action,
        amount_mid, raw_amount, is_long
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or None
    if not api_key:
        logger.info(
            "Trump PTR trades: OPENROUTER_API_KEY not set. "
            "Set it in .env to enable automated parsing of White House PTR PDFs."
        )
        return [], None

    ptr_pdfs = _scrape_whitehouse_ptrs(days_back)
    if not ptr_pdfs:
        logger.info("Trump PTR trades: no PTR PDFs found within %d days.", days_back)
        return [], None

    all_trades: list[dict] = []
    latest_date: str | None = None

    for pdf_url, date_str in ptr_pdfs:
        logger.info("Trump PTR: parsing %s  (date: %s)", pdf_url, date_str)
        trades = _parse_ptr_pdf_via_llm(pdf_url, api_key)
        logger.info("  -> extracted %d trades", len(trades))
        all_trades.extend(trades)
        if latest_date is None:
            latest_date = date_str
        time.sleep(1)   # rate-limit courtesy

    return all_trades, latest_date


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source 3 — Manual JSON fallback (non-DJT holdings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _from_manual_file(top_n: int) -> tuple[list[dict], str | None]:
    """Load holdings from data/trump_holdings.json if it exists."""
    if not _DATA_FILE.exists():
        return [], None
    try:
        with open(_DATA_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            date_str = raw.get("date")
            items    = raw.get("holdings", [])
        else:
            date_str = raw[0].get("date") if raw else None
            items    = raw
        items.sort(key=lambda x: x.get("amount", 0), reverse=True)
        holdings = [
            {
                "ticker":       str(item["ticker"]).upper().strip(),
                "company":      item.get("company", ""),
                "weight":       item.get("weight"),
                "market_value": item.get("amount", 0),
                "fund":         "Trump",
            }
            for item in items[:top_n]
            if item.get("ticker")
        ]
        logger.info(
            "Trump: loaded %d holdings from manual file (OGE Form 278e date: %s)",
            len(holdings), date_str,
        )
        return holdings, date_str
    except Exception as exc:
        logger.warning("Cannot load manual Trump holdings file: %s", exc)
        return [], None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_trump_holdings(top_n: int = 10) -> tuple[list[dict], str | None]:
    """
    Return (holdings_list, date_str) for Trump's financial disclosures.

    Strategy:
      - DJT (Trump Media) stake:  fetched live from SEC EDGAR Form 4
        (current shares × live Yahoo Finance price).
      - Other holdings (AAPL, AMZN, NVDA, etc.):  loaded from
        data/trump_holdings.json (static annual OGE Form 278e data).
      - DJT from SEC replaces/overrides the DJT entry in the JSON file
        (since the JSON may have stale price-based value).

    holdings_list: list of dicts:
        ticker, company, weight (None), market_value (USD), fund='Trump'
    date_str: most-recent source date as 'YYYY-MM-DD', or None
    """

    # Step 1 — live DJT stake from SEC Form 4
    djt_holding, djt_date = _from_sec_edgar_djt()

    # Step 2 — load manual JSON for non-DJT holdings (and as fallback for DJT)
    manual_holdings, manual_date = _from_manual_file(top_n + 5)

    # Remove any DJT entry from the manual list (we have a live one)
    if djt_holding:
        manual_holdings = [h for h in manual_holdings if h["ticker"] != "DJT"]

    # Merge: live DJT first, then manual holdings
    combined = []
    if djt_holding:
        combined.append(djt_holding)
    combined.extend(manual_holdings)

    # Sort by market_value descending, take top_n
    combined.sort(key=lambda h: (h.get("market_value") or 0), reverse=True)
    combined = combined[:top_n]

    # Pick the most informative date
    # djt_date = Form 4 filing date (when share count was last confirmed by SEC)
    # Price is always fetched live, so the market value IS current even if the Form 4 is old
    if djt_date:
        effective_date = f"{djt_date} (Form 4 share count; DJT price live)"
    else:
        effective_date = manual_date

    if not combined:
        logger.warning(
            "Trump: all sources failed.\n"
            "  -> Add holdings to data/trump_holdings.json as fallback.\n"
            "  -> SEC EDGAR Form 4 is tried automatically for DJT."
        )

    return combined, effective_date
