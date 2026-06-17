# Setup Guide

This guide sets up the local environment for reviewing and backtesting the IMC
Prosperity strategies in this folder.

## Requirements

- Python 3.11 or 3.12
- Windows PowerShell, macOS Terminal, Linux shell, or VS Code terminal
- Internet access only if packages are not already installed

Install dependencies:

```bash
pip install jsonpickle typer tqdm ipython orjson
```

The trader files themselves are intentionally lightweight. The broader
dependency list is mostly for the local backtester.

## Folder Roles

```text
.
|-- R2 - trader (final).py
|-- R1 - trader (final).py
|-- data/
|-- docs/
|-- imc-prosperity-4-backtester/
|-- datamodel.py
```

`R2 - trader (final).py` is the main strategy to review. The backtester lives in
`imc-prosperity-4-backtester/`.

## Backtester Command Pattern

Run commands from inside the backtester folder:

```bash
cd imc-prosperity-4-backtester
python -m prosperity4bt "../R2 - trader (final).py" 2 --no-vis --no-progress
```

On Windows PowerShell, either slash direction works for Python paths, but this
style is usually natural:

```powershell
cd ".\imc-prosperity-4-backtester"
python -m prosperity4bt "..\R2 - trader (final).py" 2 --no-vis --no-progress
```

## Backtesting Specific Days

```bash
python -m prosperity4bt "../R2 - trader (final).py" 2--1 --no-vis --no-progress
python -m prosperity4bt "../R2 - trader (final).py" 2-0 --no-vis --no-progress
python -m prosperity4bt "../R2 - trader (final).py" 2-1 --no-vis --no-progress
```

`2--1` means round 2, day -1. The double dash is needed because the day is
negative.

## Notes For Reviewers

- `datamodel.py` mirrors the official data classes and should not be edited
  during review.
- The backtester uses package resources under
  `imc-prosperity-4-backtester/prosperity4bt/resources/`.
- The separate top-level `data/` folder is kept as source data and for analysis
  scripts.
- The numbered strategy files are research snapshots.
- The final submission-quality files are the ones ending in `(final).py`.

## If Imports Fail

If the strategy is run directly outside the backtester and cannot find
`datamodel`, run it from this top-level project folder or add this folder to
`PYTHONPATH`.

For normal backtests through `python -m prosperity4bt`, this should not be
needed because the backtester adds the strategy's parent folder to `sys.path`.
