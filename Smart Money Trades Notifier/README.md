# Smart Money Trades Notifier

Tracks the stock portfolios and recent trades of four "smart money" sources — ARK Invest, U.S. Congress members, Trump, and Soros — and outputs everything into a single formatted Excel file.

---

## What It Does

Runs once, hits public data sources, and produces `output/smart_money_tracker.xlsx` with three sheets:

| Sheet | Contents |
|---|---|
| **Top Holdings** | Side-by-side top positions for each portfolio |
| **Recent Activity** | Individual trades (Congress + Trump PTRs) and Soros quarter-over-quarter changes |
| **Notes & Sources** | Explanation of every data source and its limitations |

---

## Project Structure

```
Smart Money Trades Notifier/
├── main.py                   # Entry point — orchestrates all scrapers, builds Excel
├── .env                      # Your configuration (API keys, settings)
├── requirements.txt          # Python dependencies
├── data/
│   └── trump_holdings.json   # Manual fallback for Trump's non-DJT stock holdings
├── scrapers/
│   ├── ark.py                # ARK ETF holdings scraper
│   ├── congress.py           # House member STOCK Act disclosures scraper
│   ├── trump.py              # Trump SEC Form 4 + White House PTR scraper
│   └── soros.py              # Soros Fund Management 13F SEC filing scraper
└── output/
    └── smart_money_tracker.xlsx   # Generated on each run
```

Only `.env.example` should be committed. Keep `.env` and generated files under `output/` local.

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your `.env` file

Copy the example file, then edit `.env` before running. The real `.env` file controls which members to track, how far back to look, and optional API keys. It is ignored by Git.

```bash
cp .env.example .env
```

```env
# Which Congress members to track (comma-separated full names)
CONGRESS_MEMBERS=Nancy Pelosi

# How many days back to look for trades
CONGRESS_DAYS_BACK=365

# Which ARK ETF to track: ARKK | ARKW | ARKG | ARKQ | ARKF | ARKX
ARK_FUND=ARKK

# Number of top holdings to show per portfolio
TOP_N=10

# Output folder
OUTPUT_DIR=output

# Required to parse Trump's White House PTR PDFs (image-scanned, needs LLM OCR)
# Free key available at https://openrouter.ai — no credit card needed for small usage
OPENROUTER_API_KEY=your_key_here

# Optional — increases OpenFIGI rate limits for Soros CUSIP lookups
OPENFIGI_API_KEY=
```

### 3. Run

```bash
python main.py
```

The output file opens at `output/smart_money_tracker.xlsx`.

---

## Command-Line Options

All `.env` settings can be overridden at the command line:

```bash
# Change number of top holdings shown
python main.py --top 15

# Track multiple Congress members
python main.py --members "Nancy Pelosi,Alexandria Ocasio-Cortez"

# Switch ARK fund
python main.py --ark ARKW

# Look back further for trades (in days)
python main.py --days 730

# Skip specific portfolios
python main.py --no-trump
python main.py --no-soros
python main.py --no-ark
```

---

## Data Sources — How Each Scraper Works

### ARK Invest (`scrapers/ark.py`)

- **Primary source:** `arkfunds.io` — a free unofficial API that aggregates ARK's daily holdings CSVs
- **Fallback:** Official `ark-funds.com` CSV download
- **Data freshness:** Updated daily (reflects previous day's close)
- **What it shows:** Current portfolio weights (% of fund) — ARKK only holds long positions

### Congress (`scrapers/congress.py`)

- **Source:** `disclosures-clerk.house.gov` — official U.S. House STOCK Act disclosure portal
- **How it works:**
  1. Downloads the annual filing index ZIP from the House clerk's server
  2. Finds all PTR (Periodic Transaction Report) filings for the target member within your look-back window
  3. Downloads each PTR PDF and parses the transactions table using `pdfplumber`
  4. Aggregates: sums purchase amounts per ticker, subtracts sale amounts → "net purchased"
- **Data freshness:** Politicians must file within 45 days of a trade — can lag real-time by up to 6 weeks
- **"Net Purchased" column:** Estimated net buy value in USD (purchases minus sales) across the look-back period. Dollar ranges in disclosures are converted to midpoints
- **Put options:** Flagged in the Recent Activity sheet as `SHORT / Bearish`

### Trump (`scrapers/trump.py`)

Trump's data comes from three separate sources combined:

**1. DJT (Trump Media) stake — live via SEC EDGAR**
- Fetches the most recent Form 4 insider filing for CIK `0000947033` (Trump's Trust)
- Multiplies the confirmed share count (currently ~114.75M shares) by a live Yahoo Finance price quote
- The Form 4 date shown reflects when the share count was last confirmed by an SEC filing — the dollar value is always calculated at today's price

**2. Other stock holdings (AAPL, AMZN, NVDA, etc.) — static JSON**
- Loaded from `data/trump_holdings.json`
- Based on the annual OGE Form 278e presidential financial disclosure
- Only updated once per year — can lag by up to 12 months
- **To update:** Edit `data/trump_holdings.json` when a new annual disclosure is published (check `oge.gov`)

**3. Recent trades — White House PTR PDFs**
- Scrapes `whitehouse.gov/disclosures/` for PTR filing PDFs
- PDFs are image-scanned (not text-based), so they are sent to an LLM via OpenRouter for OCR + extraction
- **Requires `OPENROUTER_API_KEY` in `.env`** — without it, this section is skipped
- Most of Trump's PTR trades are municipal bonds (not stocks), which is reflected accurately in the output
- Cost: ~$0.02 per PDF via `google/gemini-2.0-flash-lite-001` (the default model)
- To use a different model: set `TRUMP_PTR_MODEL=model/name` in `.env`

### Soros (`scrapers/soros.py`)

- **Source:** SEC EDGAR — CIK `0001029160` (Soros Fund Management LLC)
- **Filing type:** 13F-HR — quarterly institutional holdings disclosure
- **How it works:**
  1. Fetches the EDGAR submissions JSON to find the two most recent 13F-HR filings
  2. Downloads the `infotable.xml` from each filing (the machine-readable holdings table)
  3. Maps CUSIP codes → ticker symbols via the OpenFIGI API
  4. Computes quarter-over-quarter changes (New Position / Increased / Reduced / Closed)
- **Data freshness:** Quarterly — typically 6–8 weeks behind real-time (e.g. Q4 data available in mid-February)
- **Important limitation:** 13F filings only cover long equity positions by law. Short positions are not disclosed and are not captured here

---

## Updating Trump's Annual Holdings

When a new OGE Form 278e presidential disclosure is published (usually annually around May–June), update `data/trump_holdings.json`:

```json
{
  "_comment": "Trump's non-DJT holdings from OGE Form 278e (annual financial disclosure).",
  "_comment2": "DJT stake is fetched automatically from SEC EDGAR Form 4 — do NOT add DJT here.",
  "_comment3": "Update 'date' and 'holdings' when a new annual OGE disclosure is published.",
  "_comment4": "amount = estimated midpoint of the OGE dollar range in USD.",
  "date": "2024-05-15",
  "holdings": [
    {"ticker": "AAPL", "company": "Apple Inc", "amount": 3000000},
    {"ticker": "AMZN", "company": "Amazon.com Inc", "amount": 3000000}
  ]
}
```

- `date` — the OGE disclosure date
- `amount` — use the midpoint of the disclosed dollar range (e.g. `$1,000,001–$5,000,000` → `3000000`)
- Do **not** add a DJT entry — it is fetched live from SEC EDGAR automatically

---

## Adding More Congress Members

In `.env`:
```env
CONGRESS_MEMBERS=Nancy Pelosi,Alexandria Ocasio-Cortez,Josh Gottheimer
```

Or at runtime:
```bash
python main.py --members "Nancy Pelosi,Alexandria Ocasio-Cortez"
```

Each member gets their own columns in the Top Holdings sheet and their own rows in Recent Activity. Names must match the House disclosures index (full legal name as filed).

---

## Limitations

| Source | Limitation |
|---|---|
| ARK | Long-only fund — no short positions |
| Congress | Trades disclosed up to 45 days after the fact. Dollar amounts are ranges, not exact figures |
| Trump (DJT) | Share count from last Form 4 — only updates when the Trust buys/sells shares |
| Trump (other stocks) | Annual disclosure — may be up to 12 months out of date |
| Trump (trades) | Requires `OPENROUTER_API_KEY`. Most PTR filings are bonds, not stocks |
| Soros | Quarterly, 6–8 week lag. Long equity only — short positions not legally required to be disclosed |

---

## Disclaimer

This tool aggregates publicly available U.S. government disclosures for educational and research purposes. Nothing here is investment advice. Past trades by public figures do not predict future returns.
