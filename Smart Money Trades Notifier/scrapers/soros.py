"""
Soros Fund Management 13F Holdings Scraper
==========================================
Source: SEC EDGAR  (CIK 0001029160 — Soros Fund Management LLC)

Workflow
--------
1. Fetch the EDGAR submissions JSON to find the latest 13F-HR filing.
2. From the filing index, locate 'infotable.xml' (the holdings detail).
3. Parse the XML: each <infoTable> entry has company name, CUSIP, and value ($000s).
4. Resolve CUSIPs → ticker symbols via the OpenFIGI API (free, no key required
   for the anonymous tier; provide OPENFIGI_API_KEY in .env for higher limits).
5. Return the top-N holdings ranked by market value, with portfolio weight %.
"""

import logging
import os
import time
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

SOROS_CIK = "0001029160"
EDGAR_BASE = "https://www.sec.gov"
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{SOROS_CIK}.json"
OPENFIGI_URL    = "https://api.openfigi.com/v3/mapping"

_HEADERS = {
    "User-Agent": "SmartMoneyTracker/1.0 (contact: user@example.com)",
    "Accept-Encoding": "gzip, deflate",
}


# ── SEC helpers ───────────────────────────────────────────────────────────────

def _latest_13f_accession() -> tuple[str, str] | tuple[None, None]:
    """Return (accession_number_raw, period_of_report) for the most-recent 13F-HR."""
    try:
        r = requests.get(SUBMISSIONS_URL, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.error("Cannot fetch EDGAR submissions for Soros: %s", exc)
        return None, None

    recent = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    accessions   = recent.get("accessionNumber", [])
    periods      = recent.get("reportDate", [])

    for form, acc, period in zip(forms, accessions, periods):
        if form == "13F-HR":
            return acc, period

    logger.error("No 13F-HR found in recent EDGAR filings for Soros.")
    return None, None


def _two_latest_13f_accessions() -> list[tuple[str, str]]:
    """Return [(acc, period), (acc, period)] for the two most-recent 13F-HR filings."""
    try:
        r = requests.get(SUBMISSIONS_URL, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logger.error("Cannot fetch EDGAR submissions for Soros: %s", exc)
        return []

    recent     = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    periods    = recent.get("reportDate", [])

    results = []
    for form, acc, period in zip(forms, accessions, periods):
        if form == "13F-HR":
            results.append((acc, period))
            if len(results) == 2:
                break
    return results


def _fetch_infotable_xml(accession_raw: str) -> bytes | None:
    """Download the raw infotable.xml file from a 13F-HR filing."""
    cik_numeric = SOROS_CIK.lstrip("0")
    acc_clean   = accession_raw.replace("-", "")
    # Use the direct URL — avoids accidentally picking up the XSL-rendered HTML
    # that lives at .../xslForm13F_X02/infotable.xml
    xml_url = (
        f"{EDGAR_BASE}/Archives/edgar/data/{cik_numeric}"
        f"/{acc_clean}/infotable.xml"
    )
    try:
        xml_r = requests.get(xml_url, headers=_HEADERS, timeout=60)
        xml_r.raise_for_status()
        content_type = xml_r.headers.get("content-type", "")
        if "html" in content_type.lower():
            logger.error("Got HTML instead of XML from %s", xml_url)
            return None
        return xml_r.content
    except Exception as exc:
        logger.error("Cannot download infotable.xml: %s", exc)
        return None


# ── 13F XML parsing ───────────────────────────────────────────────────────────

_BOND_TITLE_FRAGMENTS = ("NOTE", "BOND", "DEBENTURE", "CONV", "SR NOTE", "JR NOTE")

def _is_equity(title_of_class: str) -> bool:
    """Return False for convertible notes/bonds; True for equities, ETFs, ADRs."""
    upper = title_of_class.upper()
    return not any(frag in upper for frag in _BOND_TITLE_FRAGMENTS)


def _parse_infotable(xml_bytes: bytes) -> list[dict]:
    """Parse infotable.xml into list of {name, cusip, title, value_usd} — equities only."""
    holdings = []
    try:
        root = ET.fromstring(xml_bytes)
        # Handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        for table in root.findall(f"{ns}infoTable"):
            name  = (table.findtext(f"{ns}nameOfIssuer")  or "").strip()
            cusip = (table.findtext(f"{ns}cusip")          or "").strip()
            title = (table.findtext(f"{ns}titleOfClass")   or "").strip()

            # Skip convertible bonds / notes
            if not _is_equity(title):
                continue

            # value is reported as whole dollars in modern EDGAR filings
            val_str = table.findtext(f"{ns}value") or "0"
            try:
                value_usd = int(val_str.replace(",", ""))
            except ValueError:
                value_usd = 0
            if name and value_usd > 0:
                holdings.append({"name": name, "cusip": cusip, "title": title, "value_usd": value_usd})
    except Exception as exc:
        logger.error("Error parsing infotable XML: %s", exc)
    return holdings


# ── OpenFIGI ticker lookup ────────────────────────────────────────────────────

def _cusip_to_ticker_batch(cusips: list[str], api_key: str | None = None) -> dict[str, str]:
    """
    Map a list of CUSIP strings to ticker symbols via the OpenFIGI API.
    Returns {cusip: ticker}.  Missing or failed lookups are omitted.

    Free tier: 25 requests / 10 seconds.  Each request can hold up to 10 CUSIPs.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key

    result: dict[str, str] = {}
    batch_size = 10

    for i in range(0, len(cusips), batch_size):
        batch = cusips[i : i + batch_size]
        payload = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        try:
            r = requests.post(OPENFIGI_URL, json=payload, headers=headers, timeout=20)
            if r.status_code == 429:
                logger.warning("OpenFIGI rate-limited — sleeping 12 s")
                time.sleep(12)
                r = requests.post(OPENFIGI_URL, json=payload, headers=headers, timeout=20)
            r.raise_for_status()
            resp_data = r.json()
            for cusip, item in zip(batch, resp_data):
                data_list = item.get("data", [])
                if data_list:
                    # Prefer US common equity
                    for entry in data_list:
                        if entry.get("exchCode") in ("US", "UN", "UW", "UA", "UR", "UQ"):
                            result[cusip] = entry.get("ticker", "")
                            break
                    else:
                        result[cusip] = data_list[0].get("ticker", "")
        except Exception as exc:
            logger.warning("OpenFIGI batch %d failed: %s", i // batch_size, exc)
        time.sleep(0.5)  # polite pacing

    return result


def _name_to_ticker_guess(name: str) -> str:
    """
    Last-resort: derive an approximate ticker from common company names.
    Only catches the most obvious ones; the OpenFIGI lookup is preferred.
    """
    KNOWN = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "AMAZON": "AMZN", "ALPHABET": "GOOGL",
        "NVIDIA": "NVDA", "META": "META", "TESLA": "TSLA", "BERKSHIRE": "BRK.B",
        "TAIWAN SEMICONDUCTOR": "TSM", "ASML": "ASML", "ADVANCED MICRO": "AMD",
        "BROADCOM": "AVGO", "INTEL": "INTC", "QUALCOMM": "QCOM",
        "SALESFORCE": "CRM", "NETFLIX": "NFLX", "ADOBE": "ADBE",
    }
    upper = name.upper()
    for fragment, ticker in KNOWN.items():
        if fragment in upper:
            return ticker
    # Fallback: first word, max 5 chars
    first_word = name.split()[0].upper() if name.split() else ""
    return first_word[:5]


# ── Public API ────────────────────────────────────────────────────────────────

def get_soros_holdings(top_n: int = 10) -> tuple[list[dict], str | None]:
    """
    Return (holdings_list, period_str) for Soros Fund Management's latest 13F.

    holdings_list: list of dicts:
        ticker, company, weight (%), market_value (USD), fund='Soros'
    period_str: reporting period as 'YYYY-MM-DD', or None
    """
    acc, period = _latest_13f_accession()
    if not acc:
        return [], None

    logger.info("Soros latest 13F: accession=%s  period=%s", acc, period)
    xml_bytes = _fetch_infotable_xml(acc)
    if not xml_bytes:
        return [], None

    holdings_raw = _parse_infotable(xml_bytes)
    if not holdings_raw:
        logger.error("No holdings parsed from Soros 13F XML.")
        return [], None

    # Sort by value, take top-N for ticker lookup
    holdings_raw.sort(key=lambda x: x["value_usd"], reverse=True)
    top_raw = holdings_raw[:top_n]

    total_value = sum(h["value_usd"] for h in holdings_raw)

    # Resolve CUSIPs → tickers
    api_key  = os.getenv("OPENFIGI_API_KEY") or None
    cusips   = [h["cusip"] for h in top_raw if h["cusip"]]
    ticker_map = _cusip_to_ticker_batch(cusips, api_key) if cusips else {}

    results = []
    for h in top_raw:
        ticker = ticker_map.get(h["cusip"], "") or _name_to_ticker_guess(h["name"])
        weight = round(h["value_usd"] / total_value * 100, 2) if total_value > 0 else None
        results.append(
            {
                "ticker":       ticker.strip().upper(),
                "company":      h["name"],
                "weight":       weight,
                "market_value": h["value_usd"],
                "fund":         "Soros",
            }
        )

    logger.info("Soros: %d top holdings resolved (total portfolio ~$%s M)",
                len(results), f"{total_value / 1_000_000:.0f}")
    return results, period


def get_soros_changes(top_n: int = 20) -> tuple[list[dict], str | None]:
    """
    Compare the two most-recent 13F filings and return a list of position changes.

    Each change dict has:
        ticker, company, change_type, current_value, previous_value,
        value_change, pct_change, current_period, previous_period

    change_type is one of:
        'New Position'   — appeared in latest filing, not in previous
        'Increased'      — value grew quarter-over-quarter
        'Reduced'        — value shrank quarter-over-quarter
        'Closed'         — was in previous filing, absent from latest
    """
    filings = _two_latest_13f_accessions()
    if len(filings) < 2:
        logger.warning("Soros: need at least 2 13F filings to compute changes.")
        return [], None

    (new_acc, new_period), (old_acc, old_period) = filings[0], filings[1]
    logger.info("Soros changes: comparing %s vs %s", new_period, old_period)

    new_xml = _fetch_infotable_xml(new_acc)
    old_xml = _fetch_infotable_xml(old_acc)
    if not new_xml or not old_xml:
        return [], None

    time.sleep(1)  # be polite between two EDGAR requests

    new_holdings = {h["cusip"]: h for h in _parse_infotable(new_xml)}
    old_holdings = {h["cusip"]: h for h in _parse_infotable(old_xml)}

    # Total portfolio values for weight calc
    new_total = sum(h["value_usd"] for h in new_holdings.values())

    changes: list[dict] = []
    all_cusips = set(new_holdings) | set(old_holdings)

    # Only look up tickers for the top positions by value to stay within
    # OpenFIGI free-tier rate limits (25 req / 10 s without API key).
    # Sort by max(current, previous) value and take the top 200 CUSIPs.
    api_key = os.getenv("OPENFIGI_API_KEY") or None
    top_cusips = sorted(
        all_cusips,
        key=lambda c: max(
            new_holdings.get(c, {}).get("value_usd", 0),
            old_holdings.get(c, {}).get("value_usd", 0),
        ),
        reverse=True,
    )[:200]
    ticker_map = _cusip_to_ticker_batch(top_cusips, api_key) if top_cusips else {}

    for cusip in all_cusips:
        new_h = new_holdings.get(cusip)
        old_h = old_holdings.get(cusip)
        name  = (new_h or old_h)["name"]
        ticker = ticker_map.get(cusip, "") or _name_to_ticker_guess(name)

        new_val = new_h["value_usd"] if new_h else 0
        old_val = old_h["value_usd"] if old_h else 0

        if new_val == 0 and old_val == 0:
            continue
        elif old_val == 0:
            change_type = "New Position"
        elif new_val == 0:
            change_type = "Closed"
        elif new_val > old_val:
            change_type = "Increased"
        else:
            change_type = "Reduced"

        val_change = new_val - old_val
        pct_change = (val_change / old_val * 100) if old_val else None

        changes.append({
            "ticker":           ticker.strip().upper(),
            "company":          name,
            "change_type":      change_type,
            "current_value":    new_val,
            "previous_value":   old_val,
            "value_change":     val_change,
            "pct_change":       round(pct_change, 1) if pct_change is not None else None,
            "weight_pct":       round(new_val / new_total * 100, 2) if new_total > 0 and new_val > 0 else 0.0,
            "current_period":   new_period,
            "previous_period":  old_period,
        })

    # Sort: new positions first, then by absolute value change
    priority = {"New Position": 0, "Increased": 1, "Reduced": 2, "Closed": 3}
    changes.sort(key=lambda x: (priority[x["change_type"]], -abs(x["value_change"])))

    logger.info("Soros changes: %d positions changed between %s and %s",
                len(changes), old_period, new_period)
    return changes[:top_n * 2], new_period  # return more rows since sheet 2 needs detail
