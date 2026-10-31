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
