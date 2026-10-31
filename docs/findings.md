# Model findings

## Phase 2: strength of schedule (31 Oct 2026)

Opponent adjustment (`src/ffmodel/opponent.py`) fits player level and defense-vs-position
effects together per season. Backtest 2023-2025, 3-yr weighted PPG, raw vs adjusted:

| League | QB | RB | WR | TE |
| --- | --- | --- | --- | --- |
| Burbs, avg error raw | 2.61 | 3.15 | 3.11 | 2.73 |
| Burbs, avg error adjusted (k=30) | 2.63 | 3.18 | 3.11 | 2.71 |

- Adjusting for opponents does not improve next-season predictions at any shrinkage tried (k = 10, 30, 100).
- Why: across 10+ games schedules even out. For 90% of fantasy-relevant player-seasons
  (2022-2026, 10+ PPG, 10+ games) the season schedule effect is under 0.3 PPG
  (QB 0.29, RB 0.27, WR 0.23, TE 0.13).
- Defense-vs-position effects barely repeat year to year: correlation QB 0.10, WR 0.09,
  RB 0.28, TE 0.19. Forecasting 2027 defenses from 2026 is close to guessing.
- Biggest single adjustments: J.J. McCarthy 2025 had the easiest schedule (+0.69 PPG);
  Cam Ward and Trevor Lawrence had the hardest (-0.42).

## Phase 2b: performance against tough defenses (31 Oct 2026)

Tough = the 8 hardest defenses against that position each season, judged at season end
(`src/ffmodel/tough.py`). Players 2012-2026 with 10+ PPG, 3+ tough games and 5+ other games.

- Everyone drops against tough defenses: tough-game PPG is 78% (QB, TE), 81% (RB) and 86% (WR)
  of easy-game PPG.
- Holding up against tough defenses does not repeat year to year. Correlation of the
  tough/easy ratio with next season's ratio: RB 0.05, QB 0.04, TE 0.02, WR -0.05.
- But tough-game PPG is a slightly better signal of next season's PPG for RBs (0.43 vs 0.35
  for easy-game PPG) and QBs (0.41 vs 0.38). Not for WRs (0.34 vs 0.49) or TEs (0.23 vs 0.36).
- Backtest, Burbs, 3-yr weighted PPG with tough games counted extra (avg error / rank corr):

| Tough-game weight | QB | RB | WR | TE |
| --- | --- | --- | --- | --- |
| 1.0 (baseline) | 2.61 / 0.388 | 3.17 / 0.639 | 3.11 / 0.573 | 2.73 / 0.494 |
| 1.5 | 2.63 / 0.389 | 3.15 / 0.640 | 3.07 / 0.587 | 2.80 / 0.484 |
| 2.0 | 2.69 / 0.363 | 3.16 / 0.636 | 3.05 / 0.587 | 2.87 / 0.469 |

Takeaway: a small gain for WRs and RBs, a loss for QBs and TEs. Weighting stays a tuned
per-position model setting, not a separate ranking.
