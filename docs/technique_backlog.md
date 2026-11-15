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

## To test (rough order of expected value)
0. Breakout players (Scott, Nov 15: Tuten 15.9 so far, projected 11.8): when a player's workload is far above what his preseason projection assumed for 6+ games, put less weight on the preseason number and more on his workload-based expected points. Test on 2021-25 players whose usage beat the preseason number by 50%+.
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
