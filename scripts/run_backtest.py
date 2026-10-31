"""Grade baselines (and later, models) on 2023-2025.  PYTHONPATH=src python3 scripts/run_backtest.py"""
import polars as pl

from ffmodel import backtest as bt

pl.Config.set_tbl_rows(50)
for league in ["burbs", "cbs"]:
    for name, pred in [("last season", bt.last_season(league)), ("3-yr weighted", bt.weighted_three(league))]:
        print(f"\n{league} / {name}")
        print(bt.summary(bt.grade(pred, league)))
