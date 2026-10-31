# Fantasy Football 2027 ranking model

Projects every player's 2027 stat line from his full NFL history, adjusts it for opponents,
team context (QB, play-caller, offense) and expected games played, then scores it under each
league's rules and ranks players by value over a simulated replacement level. Rankings feed the
Draft Optimizer in the auction workbook.

Leagues (settings in `config/leagues.yaml`):

| League | Draft | Starters | Scoring |
| --- | --- | --- | --- |
| The Burbs | $200 auction | QB, 2 RB, 3 WR, TE, FLEX, K, DEF | Full PPR, 4-pt pass TD, -2 INT, return yards |
| CBS 2026-2027 | Snake | QB, 2 RB, 2 WR, TE, FLEX, K, DEF | Full PPR, 6-pt pass TD, -1 INT |

## Status

Phase 1 done: data, scoring for both leagues, backtest harness. Baselines to beat (2023-2025, Burbs, avg error in PPG): QB 2.61, RB 3.17, WR 3.11, TE 2.71 (3-yr weighted PPG).

## Run

```
pip install -r requirements.txt
PYTHONPATH=src python3 scripts/build_history.py   # downloads 2012-2026, scores both leagues
python3 -m pytest -q                                # scorer reproduces nflverse PPR exactly
```

Data is cached in `data/` (git-ignored).
