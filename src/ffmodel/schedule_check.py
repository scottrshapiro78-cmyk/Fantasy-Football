"""Game-timing and injury-tag checks for advice text (Scott's rule 11, added Nov 19 2026).

Why: on Nov 19 the advice said "Walker plays tonight (Thursday)". KC played Sunday in week 11; the Thursday game was
week 12 (Thanksgiving at BUF). The day came from a pasted note, not from the schedule. Earlier, a Questionable tag
from one week was carried into the next (Tuten). Every game day, opponent or bye named in advice must come from the
nflverse schedule for the week the advice is about, and Questionable/Doubtful tags must belong to that week.

Pure functions here (testable without network); scripts/check_schedule.py loads the data and blocks on failure.
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

TEAMS = {  # abbr: words that name the team in advice text (city, nickname)
    "ARI": ["Arizona", "Cardinals"], "ATL": ["Atlanta", "Falcons"], "BAL": ["Baltimore", "Ravens"],
    "BUF": ["Buffalo", "Bills"], "CAR": ["Carolina", "Panthers"], "CHI": ["Chicago", "Bears"],
    "CIN": ["Cincinnati", "Bengals"], "CLE": ["Cleveland", "Browns"], "DAL": ["Dallas", "Cowboys"],
    "DEN": ["Denver", "Broncos"], "DET": ["Detroit", "Lions"], "GB": ["Green Bay", "Packers"],
    "HOU": ["Houston", "Texans"], "IND": ["Indianapolis", "Colts"], "JAX": ["Jacksonville", "Jaguars"],
    "KC": ["Kansas City", "Chiefs"], "LV": ["Las Vegas", "Raiders"], "LAC": ["Chargers"], "LA": ["Rams"],
    "MIA": ["Miami", "Dolphins"], "MIN": ["Minnesota", "Vikings"], "NE": ["New England", "Patriots"],
    "NO": ["New Orleans", "Saints"], "NYG": ["Giants"], "NYJ": ["Jets"], "PHI": ["Philadelphia", "Eagles"],
    "PIT": ["Pittsburgh", "Steelers"], "SF": ["San Francisco", "49ers"], "SEA": ["Seattle", "Seahawks"],
    "TB": ["Tampa Bay", "Buccaneers", "Bucs"], "TEN": ["Tennessee", "Titans"], "WAS": ["Washington", "Commanders"],
}
ALIASES = {"LAR": "LA"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TAGS = ("Questionable", "Doubtful")


def team_regex():
    """Match a team by city/nickname (any case) or by its abbreviation (upper case, whole word)."""
    out = {}
    for ab, words in TEAMS.items():
        alts = [re.escape(w) for w in words]
        out[ab] = re.compile(r"(?i:\b(?:" + "|".join(alts) + r")\b)|\b" + ab + r"\b")
    out["LA"] = re.compile(r"(?i:\bRams\b)|\bLAR?\b")
    return out


TEAM_RE = team_regex()


def week_games(sched: pd.DataFrame, week: int) -> dict:
    """{team: dict(opp, date, weekday, home)} for one week; teams missing are on bye."""
    g = {}
    for r in sched[(sched.week == week) & (sched.game_type == "REG")].itertuples():
        d = dt.date.fromisoformat(str(r.gameday))
        for t, o, h in ((r.home_team, r.away_team, True), (r.away_team, r.home_team, False)):
            g[t] = dict(opp=o, date=d, weekday=DAYS[d.weekday()], home=h)
    return g


def current_week(sched: pd.DataFrame, today: dt.date) -> int:
    """The week whose games are next (or in progress): first REG week whose last game is on or after today."""
    s = sched[sched.game_type == "REG"].copy()
    s["d"] = s.gameday.map(lambda x: dt.date.fromisoformat(str(x)))
    last = s.groupby("week").d.max()
    later = last[last >= today]
    return int(later.index.min()) if len(later) else int(last.index.max())


def player_index(teams: dict, prefer: set | None = None) -> dict:
    """{name pattern: full name}: full names always; a bare last name when it is unique among the given players,
    or unique among `prefer` (players rostered in Scott's leagues), e.g. "Walker" = Kenneth Walker III."""
    idx = {n: n for n in teams}
    last = {}
    for n in teams:
        parts = [p for p in n.split() if p not in ("Jr.", "Sr.", "II", "III", "IV")]
        last.setdefault(parts[-1], []).append(n)
    for ln, ns in last.items():
        if prefer and len(ns) > 1:
            ns = [n for n in ns if n in prefer]
        if len(ns) == 1 and ln not in idx:
            idx[ln] = ns[0]
    return idx


def _clauses(text: str):
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
        yield sent, [c for c in re.split(r";|,|\bbut\b|\bwhile\b|\bthen\b", sent) if c.strip()]


def check_text(text: str, teams: dict, sched: pd.DataFrame, today: dt.date, week: int | None = None,
               prefer: set | None = None) -> list[str]:
    """Return problems with game days, opponents and byes stated for named players.

    teams: {player full name: current team abbr}. A claim inside a clause is tied to the player named in that
    clause, or to the players named earlier in the same sentence (the claim must fit at least one); with no player,
    to the first team named. "next week", "week N" and "Thanksgiving"
    move the claim to that week; "tonight"/"today"/"tomorrow" mean today's/tomorrow's date."""
    wk0 = week or current_week(sched, today)
    idx = player_index(teams, prefer)
    name_re = re.compile(r"\b(" + "|".join(sorted(map(re.escape, idx), key=len, reverse=True)) + r")\b") if idx else None
    bad = []
    for sent, clauses in _clauses(text):
        player, wk = None, wk0
        for c in clauses:
            m = re.search(r"\bweek (\d{1,2})\b", c, re.I)
            if m:
                wk = int(m.group(1))
            elif re.search(r"\bnext week\b", c, re.I):
                wk = wk0 + 1
            names = [idx[x] for x in name_re.findall(c)] if name_re else []
            if names:
                player = list(dict.fromkeys(names))  # carried to later clauses of the sentence
            # teams named in the clause, in order of appearance
            hits = sorted((rx.search(c).start(), ab) for ab, rx in TEAM_RE.items() if rx.search(c))
            named_teams = [ab for _, ab in hits]
            cands = [(n, teams[n]) for n in (player or [])]
            subj = [x for x in cands if x[1] in named_teams]
            if subj:  # "Olave ... New Orleans plays at Chicago": the team named is the subject
                cands = subj
            elif not cands and named_teams:  # no player: "KC hosts Arizona on Sunday"
                cands = [(named_teams[0], named_teams[0])]
            if not cands:
                continue
            if re.search(r"\bthanksgiving\b", c, re.I):
                th = [w for w in sorted(sched.week.unique()) if any(
                    g["date"].month == 11 and g["weekday"] == "Thursday" and 22 <= g["date"].day <= 28
                    for g in week_games(sched, w).values())]
                if th:
                    wk = th[0]
            res = [_claims(c, who, team, wk, wk0, today, sched, named_teams, [t for _, t in cands])
                   for who, team in cands]
            if all(res):  # no named player/team fits the claim
                bad += [f"{p_}: '{sent.strip()}'" for p_ in res[0]]
    return sorted(set(bad))


def _claims(c, who, team, wk, wk0, today, sched, named_teams, own) -> list[str]:
    """Problems with the day / bye / opponent claims in clause c for one subject (player or team)."""
    out = []
    lab = f"{who} ({team})" if who != team else team
    g = week_games(sched, wk).get(team)
    neg = re.search(r"\bnot\b|n't\b|\bno game\b|\bnever\b", c, re.I)  # "Walker doesn't play tonight"
    if re.search(r"\bthanksgiving\b", c, re.I):
        if not neg and (not g or g["weekday"] != "Thursday"):
            out.append(f"{lab} does not play on Thanksgiving (week {wk})")
        return out
    rel = re.search(r"\b(tonight|today|tomorrow)\b", c, re.I)
    # a weekday used as a deadline ("after Sunday's games", "by Monday") is not a game-day claim
    day = next((m_ for m_ in re.finditer(r"\b(" + "|".join(DAYS) + r")s?\b", c)
                if not re.search(r"\b(after|before|until|by|through)\s+(this\s+|next\s+)?$", c[:m_.start()], re.I)
                and not re.match(r"'s\s+(games?|slate|kickoff)", c[m_.end():])), None)
    if rel and wk == wk0:
        want = today + dt.timedelta(days=1 if rel.group(1).lower() == "tomorrow" else 0)
        if neg and g and g["date"] == want:
            out.append(f"{lab} DOES play {rel.group(1).lower()} vs {g['opp']}")
        elif not neg and (not g or g["date"] != want):
            real = f"{g['weekday']} {g['date']:%b %d} vs {g['opp']}" if g else "bye"
            out.append(f"{lab} does not play {rel.group(1).lower()} ({want:%a %b %d}); week {wk}: {real}")
    if day and not re.search(r"\b(report|practice|news|said|says|listed)\b", c, re.I):
        if neg:
            if g and g["weekday"] == day.group(1):
                out.append(f"{lab} DOES play {day.group(1)} in week {wk}")
        elif not g:
            out.append(f"{lab} is on bye in week {wk}, not playing {day.group(1)}")
        elif g["weekday"] != day.group(1):
            out.append(f"{lab} plays {g['weekday']} {g['date']:%b %d} in week {wk}, not {day.group(1)}")
    if re.search(r"\bbye\b", c, re.I) and g and not re.search(r"\b(no|not on|off) bye\b", c, re.I):
        out.append(f"{lab} is not on bye in week {wk} (plays {g['opp']})")
    if not neg:  # every other team named must be the opponent
        for ab in named_teams:
            if ab in own or ab == ALIASES.get(team, team):
                continue
            if not g:
                out.append(f"{lab} is on bye in week {wk}, so no game vs {ab}")
            elif g["opp"] != ab:
                out.append(f"{lab} plays {g['opp']} in week {wk}, not {ab}")
    return out


def return_date(news: str, season: int = 2026):
    m = re.search(r"est\. return (\w{3}) (\d{1,2})", str(news))
    if not m or m.group(1) not in MONS:
        return None
    mon = MONS.index(m.group(1)) + 1
    return dt.date(season + (1 if mon <= 3 else 0), mon, int(m.group(2)))


def stale_tags(ros: pd.DataFrame, sched: pd.DataFrame, week: int, owned: set | None = None) -> list[str]:
    """Questionable/Doubtful tags must come from this week's report. A tag whose est. return date is before the
    current week's window (the day after the previous week's last game) was carried over from an earlier week."""
    prev = week_games(sched, week - 1)
    start = (max(g["date"] for g in prev.values()) + dt.timedelta(days=1)) if prev else None
    bad = []
    for _, r in ros.iterrows():
        if owned is not None and r["Player"] not in owned:
            continue
        if str(r["Status"]) not in TAGS:
            continue
        d = return_date(r.get("Injury News", ""))
        if d is None or start is None:
            continue
        if d < start:
            bad.append(f"stale tag: {r['Player']} ({r['Team']}) {r['Status']}, est. return {d:%b %d} is from before week "
                       f"{week} (window starts {start:%b %d}); get this week's report or clear the tag")
    return bad
