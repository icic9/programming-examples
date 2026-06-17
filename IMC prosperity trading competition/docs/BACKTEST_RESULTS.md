# Backtest Results

This file records the latest local verification run after the documentation and
comment updates.

## Command

Run from `imc-prosperity-4-backtester/`:

```powershell
python -m prosperity4bt "..\R2 - trader (final).py" 2 --no-vis --no-progress --no-out
```

## Result

```text
Round 2 day -1: 103,205
Round 2 day 0:  103,801
Round 2 day 1:  103,755
Total profit:   310,761
```

Product-level totals printed by the backtester:

```text
Day -1:
  ASH_COATED_OSMIUM:      20,806
  INTARIAN_PEPPER_ROOT:   82,399
  Total profit:          103,205

Day 0:
  INTARIAN_PEPPER_ROOT:   82,491
  ASH_COATED_OSMIUM:      21,310
  Total profit:          103,801

Day 1:
  ASH_COATED_OSMIUM:      20,669
  INTARIAN_PEPPER_ROOT:   83,086
  Total profit:          103,755
```

## Caveat

These are local backtester results on available data. They are useful for
reproducibility and sanity checking, but they should not be interpreted as a
guarantee of hidden-data performance or as a perfect model of competitor
reactions to passive quotes.
