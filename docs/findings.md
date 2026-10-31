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

## Predictor study: what predicts next-season PPG (31 Oct 2026)

3,527 player-seasons 2012-2025 (4+ games, 3+ PPG, 6+ games the next year), 38 candidate
predictors. Methods: panel OLS with player-clustered standard errors, LASSO, ridge and
LightGBM with a rolling out-of-sample test (train before t, predict t, for 2021-2025).
Full tables: `docs/predictor_study.md` (regenerate with `scripts/predictor_study.py`).

**Out-of-sample average error, The Burbs PPG (top drafted pool, 2021-2025):**

| Position | 3-yr weighted baseline | Ridge regression | LightGBM | Change, ridge vs baseline |
| --- | --- | --- | --- | --- |
| QB | 2.66 | 2.39 | 2.72 | -10% |
| WR | 2.87 | 2.54 | 2.82 | -12% |
| RB | 3.04 | 2.94 | 3.17 | -3% |
| TE | 2.47 | 2.68 | 2.78 | +8% (worse) |

**What holds up (significant in OLS and kept by LASSO):**

1. **Opportunity sticks, efficiency doesn't.** Expected fantasy points from usage repeat at
   0.55-0.77 correlation year to year, carries per game 0.67-0.93, target share 0.68-0.79 for
   RB/WR/TE. Points over expected repeat at only 0.12-0.16; yards per target 0.0-0.19; catch
   rate -0.05-0.27.
2. **TD luck reverses.** Each extra TD per game above expectation predicts -3.7 PPG next year
   for WRs and -6.6 for TEs (both p<0.001).
3. **Changing teams costs 0.7-1.4 PPG** next season at every position (QB -1.41, RB -0.99,
   WR -0.88, TE -0.65; all p<0.02), even after controlling for production and age.
4. **QB quality lifts receivers.** Each +0.1 in the team QB's EPA per attempt is worth about
   +0.43 PPG to a WR next year (p<0.001); about +0.27 for TEs (p=0.05).
5. **Age.** WRs lose about 0.27 PPG per year of age (p<0.001), RBs about 0.16 (p=0.06).
6. **Draft capital keeps mattering.** Each later draft round predicts -0.25 PPG for RBs
   (p<0.001), -0.11 to -0.12 for WRs and TEs, beyond what production explains.
7. **Two seasons beat one.** Prior-season PPG stays significant (p<=0.001 at every position)
   alongside last season's.
8. **RBs who hold up against tough run defenses gain next year** (+0.67 PPG per 1.0 of the
   tough/easy ratio, p=0.03). Not true for QB, WR or TE.

**What doesn't help:** gradient boosting (too little data per position; overfits), team
offense points per game once QB quality is known, QB-level TD luck, strength of schedule.
