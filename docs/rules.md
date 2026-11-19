# Standing rules (Scott)

Enforced by `scripts/check_rules.py` (run at the end of `scripts/weekly_refresh.py` and by `trade_finder.py`).
Add a rule here AND in check_rules.py when Scott gives a new one.

1. Keep-list (`data/input/trade_keep.txt`: Dowdle, Emmett Johnson): never offered, never dropped.
2. Never offer an injured player (Out / IR / Doubtful) in a trade.
3. News that mentions "Scott" in `data/input/news_2026.tsv` outranks the injury feed until a newer official report; the rankings must show it.
4. Add/drop advice must fit the Yahoo roster (15 spots + IR; 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 K, 1 DEF) and never leave a starting slot without a player. Check with `check_plan()` before advising.
5. Model changes (e.g. adding snap share to TE projections): adopt only if the leave-one-season-out average miss drops by at least 2% on 2018-2025 (or every season the model backtest covers), measured against the model's own projections, not a simpler points-only baseline, report the numbers to Scott first, and log the result in `docs/technique_backlog.md`.
6. Tell Scott when data is stale or missing; label guesses; the Wednesday injury report overrides unconfirmed news.
7. Before ANY add/drop advice, run `PYTHONPATH=src python3 scripts/check_rules.py plan <week> "Add1,Add2" "Drop1"`. It checks this week and next with byes and injuries counted (a player on bye or Out does not fill a starting slot), the roster cap, the keep-list and K/DEF/QB/TE availability. If it prints a problem, do not give the advice. The roster used is `burbs_rosters_2026wk<N>.csv` plus `data/input/burbs_my_moves.csv` (moves Scott has confirmed).
8. Waiver bids, add/drop and trade advice use only numbers the pipeline produced (ROS / weekly files, trade_finder, playoff_sim). A hand-applied adjustment (e.g. the Nov 17 "+2.5 Brissett bump" on Andrews) must be shown as a separate line: "untested adjustment: +X a game, based on N games", next to the pipeline number, and the bid must be sized off the pipeline number unless the adjustment has passed rule 5. (Added Nov 18 after Scott overpaid for Andrews on an untested bump.)
9. Player teams come from this season's NFL rosters. A player on no 2026 roster with no 2026 games (released/unsigned, e.g. Tyreek Hill) is dropped from the rankings, never shown under his old team.
10. Use data already on file before saying it is missing: league schedule (playoff_sim.py SCHED), standings and rosters in data/input, FAB history (Nov 16 transactions paste), league settings (config). Burbs playoff odds come from playoff_sim.py only.
11. Expert consensus SPLIT blend (Nov 19, Scott approved): src/ffmodel/expert.py. Per position, one expert weight when the model is above consensus (mostly defer) and one when below (keep the model's fades). Weekly: all four positions. ROS: RB/WR/TE; QB model only (under 2%). Edge column: Fade = model 2+ below consensus (historically right 50-76%), Model high = 2+ above (right 25-44%). Weekly only blends when that week's Friday snapshot exists (Saturday run). Preseason blend for 2027 to be refit the same way.
