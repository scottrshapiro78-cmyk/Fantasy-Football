# Standing rules (Scott)

Enforced by `scripts/check_rules.py` (run at the end of `scripts/weekly_refresh.py` and by `trade_finder.py`).
Add a rule here AND in check_rules.py when Scott gives a new one.

1. Keep-list (`data/input/trade_keep.txt`: Dowdle, Emmett Johnson): never offered, never dropped.
2. Never offer an injured player (Out / IR / Doubtful) in a trade.
3. News that mentions "Scott" in `data/input/news_2026.tsv` outranks the injury feed until a newer official report; the rankings must show it.
4. Add/drop advice must fit the Yahoo roster (15 spots + IR; 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX, 1 K, 1 DEF) and never leave a starting slot without a player. Check with `check_plan()` before advising.
5. Model changes (e.g. adding snap share to TE projections): adopt only if the leave-one-season-out average miss drops by at least 2% on 2018-2025, report the numbers to Scott first, and log the result in `docs/technique_backlog.md`.
6. Tell Scott when data is stale or missing; label guesses; the Wednesday injury report overrides unconfirmed news.
