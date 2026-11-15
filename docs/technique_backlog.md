# Technique backlog

How new prediction ideas get into the model: each one is written down here, tested on past seasons, and
adopted only if it lowers the average miss out of sample. Rejected ideas stay listed so they are not retried.
`scripts/scorecard.py` scores every week's projections after the games (data/tracking/accuracy_log.csv), and
the biggest misses each week are the main source of new ideas.

## Adopted (with the test that justified it)
- Role trend for RBs / TEs, usage luck for WRs / TEs, schedule strength (ros.fit, 2021-25)
- Healthy scratches count as zero usage, RBs (MAE 3.13 -> 2.87)
- Next man up, player-specific and shrunk to league average (2018-25, all four positions better than flat)
- QB change effect on RB / WR / TE (2019-25)
- Vacated work after a fresh long-term injury, RB 0.3 / TE 0.4
- Questionable play chances by last practice (2018-25, 2,933 cases)
- IR with no timeline: 0.3 of value after the 4-game minimum (2021-25, n=63)

## Tested and rejected
- Discounting a player's role when a teammate returns (worse MAE)
- Expert-rank blend in season (re-test set for Jan 2027 with a full season of weekly ranks)
- Vacated work for WRs (no gain)
- Less weight on the preseason number for breakout players (Nov 15 2026, scripts/study_breakout.py). 2021-25, 6+ games, workload 1.5x the preseason number and 10+ pts: 87 RBs, 63 WRs, 58 TEs. The current model was within 0.3-0.7 pts of unbiased (RB MAE 3.80); halving the preseason weight overshot by 0.9 (MAE 4.08); workload in place of the preseason number overshot by 2.8 (MAE 4.72). Same pattern at WR and TE and in 4 of 5 seasons for RBs. Breakouts fade more than their workload suggests.

## To test (rough order of expected value)
0. Expert comparison (Scott's paid rest-of-season rankings, first snapshot Nov 15 2026, scripts/compare_expert.py; the paid list itself is kept out of the repo). Rank agreement 0.83-0.91 by position. Three patterns where the model sits apart, each to test: (a) WRs stepping in for long-term injured starters: the model gives WRs no vacated work; (b) temporary starters valued as if they keep the job (a QB filling in while the starter is week to week); (c) "Out" on the report defaults to a one-week absence when news says longer. Score the expert This Week column against the model's weekly snapshot each week Scott pastes it; a blend is adopted only if it beats the model alone over 4+ weeks.
1. Play chance by injury type and position (hamstring vs ankle; QBs play 51% when Questionable vs WR 66%)
2. Weekly projection range (floor / ceiling) per player from his own volatility, for start/sit in must-win weeks
3. Role change for WRs traded mid-season (targets share on the new team after 1-2 games)
4. Backup QB play rate fitted from history instead of the flat 10%
5. Weather (wind 15+ mph, heavy rain) on QB / WR / K, from game forecasts
6. Snap share and routes run as an earlier role signal than expected points
7. Opponent pass/run funnel (defense strong one way, weak the other) rather than one number per position
8. Team pace and pass rate over expected as a matchup input
9. Offensive line injuries on RB output
10. Betting player props as a second opinion on the weekly number (needs a source the network allows)
11. Correlation between teammates in the playoff sim (QB and his WR rise and fall together)

## Schedule
- Every Tuesday run: score last week, log it, read the 20 biggest misses and tag each as injury news,
  role change, or noise. Three or more misses of the same kind in a month = new backlog item.
- First run of each month: test the top untested item and search for new published methods.
- January: full-season review, re-test everything marked rejected with the extra season.
