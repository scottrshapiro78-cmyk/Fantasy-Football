# Fantasy Football 2027 ranking model

Projects every player's 2027 stat line from his full NFL history, adjusts it for opponents,
team context (QB, play-caller, offense) and expected games played, then scores it under each
league's rules and ranks players by value over a simulated replacement level. Rankings feed the
Draft Optimizer in the auction workbook.

Leagues (settings in `config/leagues.yaml`):

| League | Draft | Starters | Scoring |
| --- | --- | --- | --- |
| The Burbs | $200 auction | QB, 2 RB, 3 WR, TE, FLEX, K, DEF | Full PPR, 4-pt pass TD, -2 INT, return yards |
| CBS 2026-2027 | Snake | QB, 2 RB, 2 WR, TE, FLEX, K, DEF | Full PPR, 6-pt pass TD, -1 INT |

## Status

Status (31 Oct 2026): all phases built. Boards: `scripts/build_boards.py`; 2026 look-back: `scripts/lookback_2026.py`. Rerun both in January on the final 2026 season. Findings: docs/findings.md.

## Run

```
pip install -r requirements.txt
PYTHONPATH=src python3 scripts/build_history.py   # downloads 2012-2026, scores both leagues
python3 -m pytest -q                                # scorer reproduces nflverse PPR exactly
```

Data is cached in `data/` (git-ignored).

## Check before sending advice: game timing and injury tags (rule 11, Nov 19 2026)

Every game day, opponent or bye named in advice must come from the nflverse schedule for the week it is about, never
from pasted text of another week, and Questionable/Doubtful tags must come from this week's report. Exit 1 = do not
send the advice until it is fixed.

```
PYTHONPATH=src python3 scripts/check_schedule.py facts "Kenneth Walker III,Chris Olave"  # write advice from this
PYTHONPATH=src python3 scripts/check_schedule.py text advice.md                         # check the draft
PYTHONPATH=src python3 scripts/check_schedule.py tags                                   # stale tags, both leagues
```

The text check ties each day ("tonight", "Thursday"), opponent ("at Buffalo", "BUF") or "bye" to the player or team
named in the same sentence, honours "next week", "week N" and "Thanksgiving", and reads "doesn't play" as a negative
claim. It cannot resolve pronouns ("his Thursday game"), so name the player. Tests: `tests/test_schedule_check.py`.
