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

## Phase 4: expected games (31 Oct 2026, corrected same day)

**Correction.** The first version counted every game a player did not appear in as missed,
so benchings, lost starting jobs and week-18 rest looked like injuries. That made Josh Allen
(16 of 17 games in 2024 and 2025) look risky (47) and inflated the claimed accuracy gain
(-9%); losing a job is easy to predict, injuries are not.

Now "missed" = weeks on reserve (IR/PUP, weekly roster status RES) or listed Out/Doubtful.
Fantasy-relevant players miss about 2-3 games a season to injury (QB 1.8, WR 2.4, RB/TE 2.7
of 17). Out of sample 2021-2025 the model misses by 2.82 games vs 2.86 for a position-and-age
average: injury history barely predicts future injuries. Only position is significant (RBs
miss most). Risk score = chance of missing 4+ games, beta-binomial with concentration 4,
calibrated: predicted 26.8% vs actual 26.9%. Among the top 150, scores run 17 to 61
(median 28); Josh Allen is 24.

## 2026 look-back, preliminary (31 Oct 2026, 8 weeks played)

`scripts/lookback_2026.py`: rankings built only from data through 2025, priced for The Burbs,
compared with the 2026 auction (142 of 180 sales matched; the rest are K/DEF and 2026
rookies, which the model cannot project yet) and with value scored so far.

Rank correlation with actual value so far (players sold, 3+ games):

| | All | QB | RB | WR | TE |
| --- | --- | --- | --- | --- | --- |
| Model $ | 0.517 | -0.02 | 0.589 | 0.417 | 0.555 |
| Room price | 0.548 | 0.23 | 0.627 | 0.509 | 0.492 |
| 30% model + 70% room | **0.569** | 0.287 | 0.640 | 0.515 | 0.510 |

The model alone does not beat the room; blended at 30% it beats both, the same weight the
2026 workbook post-mortem fitted. QB is the weak spot (calibration slope 0.5 flattens QBs).
Rerun in January when 2026 is final.

## QB projections were too low (31 Oct 2026)

Ridge regression under-projected proven QBs: for QBs with a 3-yr PPG of 20+, 2017-2025, it
projected 19.2 on average against 20.0 actual (-0.8), and Josh Allen by about 3 PPG in each of
six straight seasons. A second step (rescaling QBs by a noisy 0.5 slope) flattened them further,
which priced Allen at $8 for 2026 against his $21 sale.

Fix: QB projection = 75% ridge + 25% 3-year weighted PPG (tuned out of sample; elite bias
-0.7 -> -0.1, error 2.65 -> 2.60 Burbs), and no rescaling for QBs. Allen's 2026 look-back
projection moves 18.4 -> 20.6 PPG and $8 -> $20. Blend vs room after the fix: model 0.504,
room 0.542, 30/70 blend 0.559 (8 weeks in).

## Same bias check for RB, WR, TE and kickers (31 Oct 2026)

Proven players (3-yr PPG 16+ for RB/WR; top 10% for TE/K), 2017-2025, Burbs:

| Position | Bias before | Fix | Bias after | Avg error before -> after |
| --- | --- | --- | --- | --- |
| RB | -2.1 PPG (too low) | 40% of 3-yr history mixed in | about -0.6 | 3.08 -> 2.98 |
| WR | -0.9 PPG (too low) | 30% of 3-yr history mixed in | about -0.2 | 2.58 -> 2.62 |
| TE | +1.3 PPG (too high) | 3-yr baseline pulled 30% toward TE average | -0.2 | 2.53 -> 2.09 |
| K | +1.3 PPG (too high) | pulled 70% toward kicker average (yr-to-yr corr only 0.27) | about +0.3 | 1.30 -> about 1.0 |

## Defenses (31 Oct 2026)

DST scored for both leagues from nflverse team stats plus opponent final score as points
allowed (`src/ffmodel/dst.py`). 2013-2025, 32 teams a year, next-season DST PPG:

| Predictor | Burbs coef | p |
| --- | --- | --- |
| Last season DST PPG | +0.75 | 0.01 |
| Returning defensive snap share (0-1) | **+3.29** | **<0.001** |
| Takeaways per game (beyond PPG) | -1.28 | 0.06 |
| Def/ST TDs per game (beyond PPG) | -2.99 | 0.20 |
| Head coach change | -0.47 | 0.12 |

- **Personnel continuity is the strongest signal**: a defense bringing back 80% of its snaps
  instead of 50% projects about +1.0 PPG. It correlates with next year's DST scoring (0.30) as
  well as last year's DST scoring does (0.32).
- **Takeaways and TDs are luck** and reverse, as with skill-player TDs.
- **Head coach change** points the right way (-0.47) but isn't significant. Defensive
  coordinator changes aren't in the data and need a hand-entered 2027 list.
- **Defenses are close to unpredictable.** Out of sample 2016-2025: average error 1.69 PPG for
  the model, 2.07 for last season's PPG, 1.76 for simply projecting every defense at league
  average. Rank correlation 0.35. Supports the $3 cap and streaming defenses.

## Finished boards and the 2026 look-back on the full model (31 Oct 2026)

`scripts/build_boards.py` writes `draft_boards_<season>.xlsx` (Burbs Auction, CBS Snake,
Notes). The Burbs value format is **Max Bid (Suggested)**:
- Max bid = 30% model value + 70% the room's price for that player (or, until 2027 market
  prices are loaded, for that position slot, from the 2025-2026 auctions).
- Suggested = 90% of max, or the room's usual price if lower.
- K and DEF capped at $3. Rookies: average rookie season of past picks at the same position
  and draft range (1-10, 11-32, 33-64, 65-100, 101-160).

2026 look-back with the full model (data through 2025, room curve from the 2025 auction only),
168 sales matched, 8 weeks in. Rank correlation with value scored so far:

| | All | QB | RB | WR | TE | K | DEF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model $ | **0.565** | 0.01 | 0.53 | 0.48 | 0.64 | 0.29 | -0.07 |
| Room price | 0.558 | 0.15 | 0.61 | 0.53 | 0.49 | -0.25 | -0.23 |
| 30% model + 70% actual room price | **0.605** | | | | | | |

With rookies, kickers and defenses included, the model alone now edges the room. Mixed with
the room's real prices it is clearly better (+8%), which is why the board should be rerun with
player-specific 2027 prices in August (`build_boards.py ... 2027 market.csv`). The slot-price
version of the max bid ranks a little worse (0.544) because slot prices carry no
player-specific information; it is there for price levels, not order.

## Value upgrades 1 and 2 (31 Oct 2026)

Tested out of sample on the 2026 auction (153 matched skill players; replacement depth fitted
on the 2025 auction only; actual value measured against one fixed replacement level; 8 weeks in):

| Variant | $ error vs price paid | Rank corr with actual value | 30/70 blend with price | Avg Model $ of top 10 |
| --- | --- | --- | --- | --- |
| Current (bench-split replacement) | $7.12 | 0.519 | 0.575 | $48 |
| 1. Replacement fitted to 2025 prices | **$6.33** | 0.480 | 0.552 | $66 |
| 2. Option value (busts dropped to waivers) | $6.94 | **0.526** | **0.578** | $57 |
| 1 + 2 | $7.09 | 0.482 | 0.554 | $77 |

- **Option value is adopted** (`value(..., option=True)`, both boards). It prices each player on a
  normal range of outcomes (SD QB/RB 3.9, WR 3.5, TE 3.2, K 1.5, DEF 1.7 PPG; rookies x1.3) with
  the downside cut at replacement. Small gains on both price and ranking; stars rise.
- **Fitted replacement is not adopted.** It matches the room's price levels better (-11% error)
  but ranks players worse, because fitting to prices drops QB and TE replacement to the 10th
  player (the room underpays them), which reorders positions. Price levels already come from
  the 70% room-price share of the max bid, so the model's job is order. Fit on 2025:
  QB 10, RB 35, WR 55, TE 10 (`scripts/fit_replacement.py`). Recheck in January.

## Board format change (31 Oct 2026, Scott's call)

Max bid is now the model's value alone (walk-away price), not mixed with league prices.
League $ (what The Burbs usually pays for that position slot) sits beside it, with
Bargain $ = model value - league price and a Bargain / League overpays flag at +/-$5.
Suggested = league price, capped at 90% of the max bid.

## Role changes (Nov 2026, prompted by Walker and Tuten)
- Draft night 2026 had Kenneth Walker at RB24 ($22; room paid $40, he is RB2 through week 8) and Bhayshul Tuten at RB56 ($1; paid $20, RB15).
- Both were role changes the stats could not see: Walker signed in KC as the lead back, Tuten took over when Etienne left JAX.
- Added `open_share` and `role_gain` (share of the draft-night team's position opportunity not held by returning or arriving players). RB/WR/TE only; it hurt QBs.
- Backtest 2021-25: small gains (CBS RB rank corr 0.663 -> 0.673; others flat). Tuten moved RB56 -> RB49. Walker unchanged: KC still had backs with 2025 touches.
- Lowering the history weight for big role changes helped WR, hurt RB: not adopted.
- 2026 through week 8 (top players, 3+ games): Rotoballer preseason projections beat the model on rank at QB (0.46 vs 0.09), RB (0.53 vs 0.40), WR (0.59 vs 0.54); the model wins at TE (0.88 vs 0.61). A 70% Rotoballer / 30% model blend is about as good as Rotoballer alone at RB and best at WR. One partial season only.

## Draft-night news as data (Nov 2026: Walker, Tuten, Jacobs)
- Added draft-night depth chart rank (week-1 chart through 2024; ESPN snapshot on or before Aug 31 from 2025), contract size (% of cap) and "paid to join a new team" from OverTheCap contracts.
- Added known suspensions: consecutive weeks from week 1 on Reserve/Suspended (R40) or exempt list, removed from expected games. For a future season, list announced suspensions in data/input/suspensions_<year>.csv (Player, Games).
- Backtest 2021-25 (Burbs MAE / rank corr): QB 2.55 -> 2.44; RB 2.97 -> 2.90, 0.661 -> 0.677; WR and TE flat.
- 2026 draft night through week 8 (rank corr with actual PPG): QB 0.09 -> 0.13, RB 0.40 -> 0.45, WR 0.54 -> 0.62 (Rotoballer 0.61), TE 0.88. Rotoballer still better at QB (0.46) and RB (0.53).
- Walker RB24 $22 -> RB17 $27 (room paid $40). Tuten RB56 $1 -> RB45 $7 ($20). Jacobs now carries 6 games out (6.3 of 13 expected games).
- 2027 boards: depth charts and suspensions fill in once August data exists; until then those features sit at neutral values.

## Rest of season (Nov 2026)
- ROS PPG = (k x preseason + n x so-far) / (k + n); k fitted at week 8 on 2021-25: QB 5-6, RB 5-6, WR 3, TE 2-3, K 20, DEF 30.
- Role trend (usage-based expected points, last 3 games minus season) helps RB and TE a little (Burbs RB 3.05 -> 3.01, TE 2.52 -> 2.47); not QB/WR.
- Cutting the preseason weight when this season's usage differs a lot from the projection hurt RBs (usage spikes regress); small help for QBs; not adopted.
- Remaining schedule (defense vs position from this season's games, backfitted against each player's own level): DEF 2.16 -> 2.06, TE 2.47 -> 2.44, RB/WR ~0, QB 0. Effects for skill players are under ~0.3 PPG after shrinkage.
