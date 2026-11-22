# Answer bank: Scott's questions already studied (read this FIRST, rule 13)

One entry per question. Reuse the answer; rerun the script only if Scott asks for fresh numbers or the
season's data has changed enough to matter (say so when reusing: "from the Nov 21 study").
Format: question -> answer (data, n) -> script. Newest at the bottom of each section.
When a later answer corrects or sharpens one, EDIT the entry in place (keep a one-line "was: ..." note) so the bank always holds the best answer.
Model-wide study results (schedule strength, tough defenses, refits, techniques tried) live in docs/findings.md and docs/technique_backlog.md; trade/lineup answers below go stale when rosters change: rerun those, don't reuse the numbers.

## Injuries and availability
- **Will a player on the injury report play? (status, practice, history, team record)** Nov 21, 2009-2025 (injury reports start 2009), 9,056 player-weeks.
  Played rate: Out/Doubtful 0-2%; Q+DNP 44%; Q+Limited 75%; Q+Full 88%; not listed+DNP 85%; not listed+Lim/Full 97-98%.
  Q players: missed a game in prior 8 wks 60-65% vs 77%; hamstring 58%, calf 62%, quad 64%, ankle 72%, knee 75%, groin 79%, back 92%.
  Team record: no effect in-season (73/72/73/68% worst->best). Q+Lim, listed 4+ of last 8 wks, never missed: 88% (n=144).
  Logistic with history+team: 8% lower Brier than status table (0.0605 vs 0.0656), well calibrated. NOT wired into Play Chance (needs rule 5 test).
  (was: 2018-2025 only, n=4,237; same conclusions.) -> scripts/study_play_given_status.py
- **Do teams out of contention sit stars with nagging injuries? (weeks 14-17)** Nov 19 (other thread, branch claude/project-thread-mxt536).
  Share of games missed: healthy+alive 6%, healthy+dead 11%, nagged+alive 12%, nagged+dead 18%; week 18 nagged+dead 34%.
  Effects add, ~1 pt/wk for an 18-PPG star: not adopted. -> scripts/study_shutdowns.py (that branch)
- **Questionable TE/WR who plays: full snaps or decoy?** Nov 21, 2012-2025 (snap counts start 2012), starters (50%+ snaps), vs own last-4.
  TE ankle (n=64): snaps 98% of norm, 25% lose 20%+ snaps (healthy 17%), -0.7 targets, -2.0 pts on 9.4 (~-20%).
  TE Q+Limited ankle (n=45): snaps 93%, 31% big drop, -1.9 pts. WR ankle (n=132): snaps 101%, -0.6 tgt, -1.7 pts on 12.4.
  Not a decoy; modestly less productive. (was: 2018-2025, TE ankle n=27, -2.5 pts.) -> scripts/study_q_ankle_snaps.py
- **Player-specific in-season injury odds (tag, returning, age, workload)** -> scripts/study_injury_conditional.py (see technique_backlog.md)

## QB changes
- **Does a new QB lift his TE/WRs? (Brissett/Andrews)** Nov 18: 359 QB changes since 2021, no reliable TE lift; rejected (technique_backlog.md).
  Nov 21 fact check: Andrews with Lamar/Brissett 5 g 11.0 PPG; with Huntley/Fagnano 5 g 4.0 PPG. The drag is bad backups,
  not a Brissett boost. QB Context column in the weekly file shows this for every RB/WR/TE. -> scripts/check_rules.py facts "<player>"

## Experts and projections
- **Expert consensus blend** (FantasyPros ECR, split weights): rule 11, technique_backlog.md. Model above consensus by 2+:
  model right 25-44%; below: 50-60% (fades 76%). Footballguys/RotoBaller side with FantasyPros on model fades 13/15, 14/14.
- **Yahoo vs model, week 11** Nov 20: correlation 0.93 after the stale-tag fix (0.88 before); Yahoo runs ~1 pt higher;
  biggest gaps are Questionable players (model multiplies by play chance, Yahoo does not). File: inputs/expert/yahoo_weekly_2026wk11_burbs.csv
- **Why is X ranked WR20 / why does the model love Odunze?** Rank within the week depends on byes; check Consensus and
  Model - Consensus in the weekly file before explaining. A gap under 2 pts is not an Edge call.

## Playoff odds (Burbs, Gin and Jews) - rerun only when standings, rosters or injuries change
- Nov 21 (after final injury report): playoffs 93%, title 23%; week 11 vs Broken Ankles 128.2-130.6 (46%).
  Win: playoffs 98%, title 26%, bye 30%. Loss: 88%, 20%, 6%. CBS: playoffs 46%; win 48%, loss 17%.
  -> scripts/playoff_sim.py (prints title/bye by this week's result)

## Lineup / roster mechanics
- Yahoo locks each player at his own kickoff; a bench player whose game started cannot replace a late scratch.
  A backup for a Sunday-late player must play later (Monday night). (Nov 21, Gesicki as Bowers' fallback.)
- Roster moves go in data/input/burbs_my_moves.csv (add / drop / ir). IR slot does not count against 15.

## Kickers
- **ROS kicker rankings from offensive opportunities; does defense strength matter?** Nov 18, 2021-2025 games.
  Weak-third defenses 8.4 vs strong-third 7.7 PPG, but adding defense strength did not improve out-of-sample error (+0.1%).
  Scoring chances (drives to the 30, red-zone stalls, PATs) 0.25% better; indoor +1.0 / home +0.4 pts, 0.6% better; cold December: none.
  Nothing wired in. Bass ranked 8th of 32 (wks 11-17); keep him. Avoid unsettled-K teams (MIA, LA, NO, BAL, ATL at the time).
  -> branch claude/project-thread-3op59e, /mnt/project-files/ff-model/kickers/

## Teammates and targets
- **When Olave has a strong WR2, do his targets drop?** Nov 19, 2022-2026 wk10. Strong WR2 = teammate 18%+ target share.
  With: 25 g, 8.4 tgt/g, 25.3% share; without: 39 g, 8.6, 26.1%. No clear effect (small sample, QB/play-caller changes).
- **Low projection for a WR whose teammates were out (Lemon, Nov 19):** Lemon with Smith and Wicks both playing 3.8 PPG on 4 tgt;
  with either out 11.7 on 8 tgt. A stale Scott note (both Out) had inflated him; notes now expire (check_schedule.py notes).
  General lesson: a breakout during teammates' absences fades when they return; check absences before calling a breakout.

## Trades (numbers go stale when rosters change; rerun trade_finder / the thread's method)
- **Tuten 1-for-2 (WR + flex RB), Burbs, Nov 14:** best was Barkley + Tate (MTAT) ~+1.5/wk; without MTAT, Garrett Wilson + Woody Marks ~+1.0/wk.
  Gains were small because the starters were already strong. Scott kept Tuten.
- **CBS RB-for-WR with No-Huddle Brent, Nov 19 (Johnson locked, rule 1):** Hampton + Lemon for Olave +6.3/+5.7 per wk (wks 12-14 / playoffs) was the
  best realistic offer after the split blend; Walker + Lemon for Smith-Njigba +5.6/+5.5; Hampton + G. Wilson for Smith-Njigba +11.9/+10.4 (long shot).
  Don't trade with upcoming opponents or playoff rivals (Herb wk14, Nick Viz wk12, Kline wk13). Process trades after the player's game if the incoming player is on bye.

## Mechanics learned the hard way
- Questionable tags apply only to the next game; never discount a player weeks ahead (Nov 18 Tuten week 13 bug).
- Game days/opponents in advice must come from the current-week nflverse schedule, not pasted text (Nov 19 Walker "plays tonight" error; check_schedule.py).
- Win % vs Yahoo's: same projected totals, Yahoo's percentage is flatter; ours uses SD 21 per team.
