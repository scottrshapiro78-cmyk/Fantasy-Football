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
- Injury return dates from ESPN's injury page on every run (Nov 15 2026); fill-in QBs valued only until the regular starter is due back
- RB depth-chart moves and short fill-ins (Nov 15 2026): RBs who lost the lead job while healthy x0.85 (n=21, miss 2.24 -> 1.88); new lead backs get a note only (n=52, model already unbiased: 10.1 vs 10.0); the next RB / TE on the depth chart gets his own fill-in bump for the 1-2 games a lead is out, spread into ROS PPG
- WR depth-chart moves (scripts/study_new_starter.py, 2021-25): new starters +0.75 (model ran 0.9 low, n=119); WRs dropped from the starters x0.85 (model ran 1.3 high, n=47, miss 3.16 -> 2.9)

## Tested and rejected
- Discounting a player's role when a teammate returns (worse MAE)
- Expert-rank blend in season (re-test set for Jan 2027 with a full season of weekly ranks)
- Preseason: team-down correction from last season's team scoring or last season's Vegas implied points (Nov 15 2026, scripts/study_preseason.py, 2021-25 out of sample, drafted pool n=694): miss 2.76 -> 2.78, no gain at any position
- Preseason: player-specific boom/bust ranges from age, experience, team change, games missed, TD luck: predicted width had zero correlation with the actual miss (-0.11 to 0.02). A one-size 80% range per position (about +/- 4 to 5 PPG) is as good
- Vacated work for WRs (no gain)
- Projecting every new WR starter at a starter's level (floor of 6-9 PPG): miss rose from 3.05 to 3.26-3.68. New starters split: a quarter finish under 4.3 PPG, a quarter over 10.7
- Less weight on the preseason number for breakout players (Nov 15 2026, scripts/study_breakout.py). 2021-25, 6+ games, workload 1.5x the preseason number and 10+ pts: 87 RBs, 63 WRs, 58 TEs. The current model was within 0.3-0.7 pts of unbiased (RB MAE 3.80); halving the preseason weight overshot by 0.9 (MAE 4.08); workload in place of the preseason number overshot by 2.8 (MAE 4.72). Same pattern at WR and TE and in 4 of 5 seasons for RBs. Breakouts fade more than their workload suggests.

## To test (rough order of expected value)
0. Expert comparison (Scott's paid rest-of-season rankings, first snapshot Nov 15 2026, scripts/compare_expert.py; the paid list itself is kept out of the repo). Rank agreement 0.83-0.91 by position. Three patterns where the model sits apart, each to test: (a) WRs stepping in for long-term injured starters: the model gives WRs no vacated work; (b) temporary starters valued as if they keep the job (a QB filling in while the starter is week to week); (c) "Out" on the report defaults to a one-week absence when news says longer. Score the expert This Week column against the model's weekly snapshot each week Scott pastes it; a blend is adopted only if it beats the model alone over 4+ weeks.
0b. Preseason Vegas team totals for QBs (August 2027). Best case using the season's own game lines (not knowable in August): QB miss 2.68 -> 2.36 (-12%), RB 2.92 -> 2.87, WR and TE nothing, all positions 2.76 -> 2.69. Preseason win totals / team point totals would capture part of that. Needs the August sportsbook team totals (Scott to paste, or a source the network allows); apply to QBs only, about +0.3 to 0.5 PPG per implied point above average.
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

## Tested Nov 15 2026: known-in-August team changes (scripts/study_team_changes.py)
2021-2026 out of sample (2026 = 8 games), drafted pool, burbs scoring. Flags: new head coach (47 team-seasons),
new week-1 QB (62), QB back from injury (20), offensive line turnover (share of last year's line snaps gone; avg 28%).
Average miss with a correction fitted on earlier seasons: model 2.80; +new_hc 2.80; +new_qb 2.81; +qb_back 2.80;
+ol_turn 2.79; all together 2.83; Vegas best case 2.71; Vegas + all flags 2.75. REJECTED: none beats the model, and
adding them to Vegas makes Vegas worse. Raw patterns seen but not reliable out of sample: QBs starting for a new team
missed by -2.3 PPG (n=17); WRs on top-third line-turnover teams -1.3 vs -0.4 on bottom third.
NOT TESTED: offensive line coach and play-caller changes. No reliable history source reachable (Wikipedia season
pages fetched for 2020-26 did not return the staff boxes). Needs a team-by-season list (season,team,new_oc,new_ol_coach)
in data/input/coach_changes.csv; the script picks it up automatically.

## Tested Nov 16 2026: player-specific in-season injury odds (scripts/study_injury_conditional.py)
10,384 at-risk player-weeks, weeks 9-17, 2018-25 (played week w-1; event = misses week w). Flat rate 7.8%.
In-sample groups: clean (no tag, no missed games) QB 6.0 / RB 6.9 / WR 6.0 / TE 7.2%; played through an injury tag
6.8 / 9.2 / 12.0 / 13.1%; just back from a missed game 11.1 / 11.5 / 14.4 / 18.5%. Age, heavy workload and last
season's missed games: no reliable effect (heavy-workload players miss slightly LESS).
Out of sample (fit 2018-23, score 2024-25): Brier flat vs model QB .0731/.0721, RB .0490/.0488, WR .0548/.0548,
TE .0818/.0818. Rest-of-season share of weeks 10-17 missed: MAE flat vs model QB .215/.196, RB .161/.158,
WR .169/.161, TE .189/.192; only QBs separate (missed time earlier: 34% vs 12%, mostly job security).
REJECTED for the sim: signals are short-lived (next week or two) and the weekly rankings already use the current
injury report; no reliable rest-of-season gain. Keep the flat position rate.
