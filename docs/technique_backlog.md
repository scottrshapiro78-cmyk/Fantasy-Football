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
- Weekly QB formula refit (Nov 26 2026, scripts/study_consensus_misses.py): Vegas 0.45 vs team's usual + 0.21 vs league, matchup 0.09, Base PPG x0.66, Questionable -0.8, back from a missed game -6. Miss 6.23 -> 6.09, start/sit disagreements with consensus right 45% -> 54% (2+ pts)
- WR depth-chart moves (scripts/study_new_starter.py, 2021-25): new starters +0.75 (model ran 0.9 low, n=119); WRs dropped from the starters x0.85 (model ran 1.3 high, n=47, miss 3.16 -> 2.9)

## Tested and rejected
- TE targets per game and TE snap share in rest-of-season projections (Nov 18-19 2026, scripts/study_te_targets.py).
  Nov 18 on 2019-25 (n=421, data from 2012): vs the model, targets +0.6% (worse), snaps -0.2%. REJECTED then.
  Nov 19 long run (FF_DATA=data_2006 FF_FIRST_SEASON=2006: data 2006-26, model backtest 2015-25, n=650; 2013-14 fail
  because the DST model needs snap data from 2013): model as-is 2.165, model refit 2.202; +targets 2.183 (-0.9% vs refit,
  REJECTED); +snap share 2.129 (-3.3% vs refit, -1.7% vs as-is; better in 8 of 11 seasons). Then fitted INSIDE the model
  (k/u/r/s grids plus a snap weight, leave-one-season-out, 2015-25, n=650): 2.167 without snaps vs 2.168 with (+0.1%).
  The earlier -3.3% came from the straight-line recalibration, not from snaps. REJECTED. Gesicki's 9.2 stands.
- ROS value: steadier replacement-level anchor (the anchor's per-game option value x median games of the five players around replacement, so one injured player at replacement can't shift a position; Nov 16 2026, scripts/study_replacement_anchor.py, 2021-25 at weeks 8-11, 20 cases per league): cross-position rank correlation with actual ROS value fell (Burbs 0.660 -> 0.646, CBS 0.673 -> 0.663; better in only 8 and 7 of 20 cases), RB error rose (Burbs 11.5 -> 13.4). Week-to-week steadier (RB top-12 jump 7.2 -> 6.4) but less accurate. Kept as value(robust_base=False)
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

## Added Nov 17 2026 (from live mistakes; test before adopting, report to Scott first)
- QB -> TE/WR target lift: pass-catchers' targets and points by QB (Brissett TE usage; Cousins +4.5 for Jefferson). Backtest by QB identity, shrunk by games.
- Backup uplift from the player's own games without the starter, shrunk toward the league TE2 +1.3 / RB2 average.
- Playoff-week availability by team elimination (shutdown risk): eliminated teams' lead RB plays 74% of weeks 15-17 vs 87%; backups do not gain (tested Nov 17).
- Snap share for TEs: tested, +2% only, not adopted.

## QB change -> TE/WR/RB effect (tested Nov 18, scripts/study_qb_change.py)
359 QB-change events since 2021 (79 TE, 174 WR, 106 RB), leave-one-season-out. Feature: change in the QB's TE (or WR) target share x the player's share of his position.
TE: -1% to -4% MAE (worse than no adjustment). WR: +0.4% to +1% (lead WRs +5% MAE, tiny sample). RB with QB carries: -0.5%. None clears the 2% rule: REJECTED, not wired in.
Brissett TE share 0.30 vs Lamar 0.31 (league 0.22): by this spec no lift for Andrews.
Restricted rerun (Nov 18): TE only when incoming QB TE share > league (n=30): -2.1% MAE; and up vs old QB (n=14): -11%. RB only when incoming QB carries > league avg 3.4/g (n=39): -1.1%; and up vs old (n=18): +0.7%. Still rejected.

## Longer data (2006+), all positions (tested Nov 19, scripts/study_long_data.py, study_weekly_long.py, study_waivers.py)
- ROS (after week 10, 2015-25 backtest): refitting k/u/r/s on 2015-25 vs production (fit 2021-25): QB +0.3%, RB -0.5%, WR +0.2%, TE +0.5%.
  Extra terms: snap share 0 to +0.5% worse; WR/TE targets per game, RB touches: no gain. REJECTED, production weights stand.
- Preseason, graded 2019-25: ridge trained 2012+ (production) QB 2.53 / RB 3.00 / WR 2.70; ridge 2006+ 2.58 / 2.98 / 2.74;
  LightGBM 2006+ 2.67 / 3.06 / 2.77. Older seasons do not help (only RB -0.6%); LightGBM worse everywhere. Keep ridge 2012+.
- Weekly (2006-25, simplified replication): production matchup/Vegas weights are 0.6-1.0% worse than no adjustment.
  Refit (much less matchup, more Vegas: QB 0.30/0.225, RB 0.05/0.25, WR 0.10/0.10, TE 0.05/0.125) beats production by
  0.7-1.65%: under the 2% bar. Not adopted; candidate if Scott wants it.
- Waivers (2006-26, undrafted pool, decision weeks 3-12, top 3 per position): season-to-date PPG is the best single pick
  signal (hit rate RB 75%, WR 63%, TE 64%, QB 52%); last-3 next; last-week points and usage jumps are the worst
  (usage jump hit rate ~30%). A ridge blend of all signals adds 0-1 point of hit rate. No change to add/drop logic.

## Expert consensus blend (tested Nov 19, scripts/study_expert_blend.py) - ADOPTED Nov 19 where it passes (Scott)
FantasyPros consensus rankings (ECR), free archive: raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr.parquet
(weekly snapshots late 2020 to now, updated weekly). ECR is a rank, so the expert order gets the model's own sorted numbers;
blend weight w picked leaving the graded season out. MAE model -> blend:
- Preseason 2021-25: QB 2.75 -> 2.44 (-11.2%, w 1.0), RB 2.91 -> 2.81 (-3.4%, w .5), WR 2.77 -> 2.64 (-4.6%, w .9), TE 2.36 -> 2.22 (-6.1%, w .9)
- ROS after wk10, 2020-25: QB -2.1% (w 1.0), RB -2.9% (w .65), WR -1.7% (w .55, under bar), TE -3.2% (w .9)
- Weekly wks 5-17, 2020-26 (simplified weekly model): QB -4.5%, RB -4.2%, WR -3.9%, TE -2.8% (w .8-1.0)
Caveats: weekly base is the simplified replication, not build_weekly itself; ROS snapshot can include week-11 injury news the backtest model lacks.
- Retest vs the real weekly model (base = ROS formula from ros_history_wk9-11, predicting weeks 10-12, 2021-25, + production
  matchup/Vegas): QB -5.4% (w 1.0), RB -2.9% (w .9), WR -3.5% (w 1.0), TE -0.1% (rejected). Wired in: weekly QB/RB/WR, ROS QB/RB/TE.
- Note: ros_history_wk{t} files hold games THROUGH week t (not t-1). study_long_data's ROS grading counted week t in both "so far"
  and "after"; same for every variant, so the comparisons stand, but absolute MAEs there are slightly optimistic.
- Disagreement study (Nov 19): gap 2+ PPG, model HIGHER than consensus -> model closer only 25-44% (weekly/ROS/preseason);
  model LOWER -> 50-60% (slump fades 76%, below-hype breakouts 65%; role-up + model higher only 33%).
  Split blend (expert weight by direction, LOSO) vs same-weight blend: ROS RB -3.9% vs -2.8%, WR -2.8% vs -1.7%, TE -3.3% vs -2.8%;
  weekly RB -3.9% vs -2.9%, WR -3.8% vs -3.5%, TE -3.2% vs -0.1%; QB no gain (experts better both ways). ADOPTED Nov 19 (Scott), with an Edge column.

## QB QUALITY adjustment (tested Nov 21, scripts/study_qb_quality.py) - REJECTED
Pre-registered (rule 14). QB quality = career passing fantasy pts/attempt to date (2006+ and 2012+ ratings; sanity: Lamar +24%, Mahomes +17%,
Brissett -14%, Huntley -27% vs league). Model's own week-11 ROS projections 2015-2025, RB/WR/TE, 2,729 player-seasons (~19% had a QB change).
V1 current QB vs this season's QBs: -0.2% / -0.1% (worse), 2-3 of 10 seasons better. V2 + last season's QBs: 0.0% / +0.1%, 2025 holdout -0.6%.
V3 ORACLE (QBs he actually had after the cut): -0.1% / -0.1%. Even perfect knowledge of future QB quality does not cut the miss:
the model's role and recent-usage terms already absorb it (targets/xFP move with the QB). On QB-changed players only: 3.36 -> 3.36-3.45.

## Why the model loses its arguments with the experts (tested Nov 26, scripts/study_consensus_misses.py) - QB ADOPTED
Trigger: week 12 Dak 20.6 vs Lawrence 19.1 while every source had Lawrence ahead (JAX implied 31.5 vs DAL 29, Dak 29.9 last 3).
Data: 2021-2026 weeks 5-17 (2026 through wk11), fantasy-relevant pool per week (top QB20/RB40/WR50/TE18 by model or by
FantasyPros), Burbs scoring, base = the production ROS formula at t-1. Leave-one-season-out median regression.
FantasyPros used ONLY to grade: same-position pairs where the model's order and the consensus order differ, who was right.
- Before: when the model and consensus disagreed on a pair the model was right 43% (QB), 43% (RB), 44% (WR), 46% (TE);
  with the model's margin 2+ points, 45-49%. 2026 QBs: 35% (17 of 48).
- Hypotheses: (1) too much recent form: NO. QBs have no last-3 term; a last-3 term gets a small POSITIVE weight (+0.13).
  Dak's number came from his season average, not his streak. (2) Vegas ignored: YES for QBs. Fitted 0.41-0.45 per implied
  point above the team's usual + 0.21-0.25 per point above the league average (production: 0). (3) Matchup over-weighted for
  QBs: YES, fitted 0.09-0.17 vs production 1.0 (the line already prices the defense). (4) Shrinkage: YES, Base PPG fits at a
  slope of 0.64-0.66 for QBs (0.74-0.88 other positions): weekly spread is narrower than the base implies.
  (5) Pre-game injury facts: Questionable-and-plays -0.4 to -0.8, back from a missed game -5.8 (QB, n=56), -1.2 RB, -1.4 WR.
- QB (vegas + shrink + injury tags): miss 6.23 -> 6.09 (-2.3%, 5 of 6 seasons), disagreements right 43% -> 50%, at 2+ points
  45% -> 54%, 2026 35% -> 71%. Model alone now equals consensus on QBs (6.09 vs 6.07 on the model's scale). ADOPTED
  (build_weekly QB_CAL, coefficients fitted on all six seasons). Week 12 rerun: Lawrence 22.4, Dak 20.9.
- RB -0.4%, WR -2.0%, TE 0.0% with the same terms; adding fresh teammate absences RB -1.3% / WR -2.2% / TE +0.1%, and RB
  disagreements 43% -> 46% (2+ points 47% -> 50%). Not adopted: under the 2% rule or overlaps the existing Teammate Out Adj
  (production's fill-in bumps, which this backtest base does not include). RB/WR "model high" calls stay weak (33-39% right):
  the experts know about role changes the box scores don't show yet; usage share last game/last 3 did not help.
- Andrews (wk5-11): the model was BELOW consensus every week (8-9 vs TE13-17) and he scored 1.4-5.3: those fades were right.
