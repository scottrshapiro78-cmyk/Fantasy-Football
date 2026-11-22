# Answer bank: Scott's questions already studied (read this FIRST, rule 13)

One entry per question. Reuse the answer; rerun the script only if Scott asks for fresh numbers or the
season's data has changed enough to matter (say so when reusing: "from the Nov 21 study").
Format: question -> answer (data, n) -> script. Newest at the bottom of each section.

## Injuries and availability
- **Will a player on the injury report play? (status, practice, history, team record)** Nov 21, 2018-2025, 4,237 player-weeks.
  Played rate: Out/Doubtful ~0%; Q+DNP 50%; Q+Limited 72%; Q+Full 87%; not listed+DNP 85%; not listed+Lim/Full 97-98%.
  Q players: missed a game in prior 8 wks 63% vs 76%; hamstring 55%, quad 62%, shoulder 65%, ankle 73%, knee 75%, groin 84%.
  Team record: no effect in-season (70/73/72/66% worst->best). Q+Lim, listed 4+ of last 8 wks, never missed: 88% (n=65).
  Logistic with history+team: 7% lower Brier than status table, well calibrated. NOT wired into Play Chance (needs rule 5 test).
  -> scripts/study_play_given_status.py
- **Do teams out of contention sit stars with nagging injuries? (weeks 14-17)** Nov 19 (other thread, branch claude/project-thread-mxt536).
  Share of games missed: healthy+alive 6%, healthy+dead 11%, nagged+alive 12%, nagged+dead 18%; week 18 nagged+dead 34%.
  Effects add, ~1 pt/wk for an 18-PPG star: not adopted. -> scripts/study_shutdowns.py (that branch)
- **Questionable TE/WR who plays: full snaps or decoy?** Nov 21, 2018-2025, starters (50%+ snaps), vs own last-4.
  TE ankle (n=27): snaps 101% of norm, 22% lose 20%+ snaps (healthy 17%), -0.7 targets, -2.5 pts on 9.2 (~-25%).
  TE Q+Limited ankle (n=20): snaps 95%, 30% big drop, -2.5 pts. WR ankle (n=81): snaps 100%, -0.8 tgt, -2.6 pts on 12.9.
  Non-leg injuries cost less (-1.2 TE, -1.7 WR). -> scripts/study_q_ankle_snaps.py
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
