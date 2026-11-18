"""Burbs roster ledger: the latest pasted rosters file plus confirmed moves since then.

data/input/burbs_my_moves.csv: action,Player,Pos[,Team]  (Team defaults to Scott's "Gin and Jews").
Every script that reads Burbs rosters (check_rules, trade_finder, playoff_sim) goes through apply(), so a
confirmed move is entered once and used everywhere. A newer pasted rosters file replaces the ledger: clear it then.
"""
import pandas as pd

from ffmodel.config import DATA

ME = "Gin and Jews"


def moves() -> pd.DataFrame:
    p = DATA / "input" / "burbs_my_moves.csv"
    if not p.exists():
        return pd.DataFrame(columns=["action", "Player", "Pos", "Team"])
    m = pd.read_csv(p).fillna("")
    if "Team" not in m.columns:
        m["Team"] = ""
    m["Team"] = m["Team"].replace("", ME)
    return m


def apply(roster: pd.DataFrame) -> pd.DataFrame:
    """roster: Team,Slot,Player,... (a pasted Burbs rosters file). Returns it with the ledger moves applied."""
    r = roster.copy()
    for m in moves().itertuples():
        if m.action == "drop":
            r = r[~((r.Team == m.Team) & (r.Player == m.Player))]
        elif m.action == "add":
            r = r[r.Player != m.Player]   # a player can only be on one team
            row = {c: "" for c in r.columns}
            row.update({"Team": m.Team, "Slot": "BN", "Player": m.Player})
            r = pd.concat([r, pd.DataFrame([row])], ignore_index=True)
    return r.reset_index(drop=True)
