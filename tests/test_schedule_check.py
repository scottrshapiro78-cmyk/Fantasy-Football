"""Rule 11 checks on a small made-up schedule (no network). Mirrors the Nov 19 2026 Walker mistake."""
import datetime as dt

import pandas as pd

from ffmodel.schedule_check import check_text, current_week, stale_tags

SCHED = pd.DataFrame([
    # week 10
    dict(week=10, game_type="REG", gameday="2026-11-15", home_team="KC", away_team="DEN"),
    dict(week=10, game_type="REG", gameday="2026-11-16", home_team="SEA", away_team="LA"),
    # week 11: IND-HOU on Thursday, KC and NO Sunday, SEA on bye
    dict(week=11, game_type="REG", gameday="2026-11-19", home_team="HOU", away_team="IND"),
    dict(week=11, game_type="REG", gameday="2026-11-22", home_team="KC", away_team="ARI"),
    dict(week=11, game_type="REG", gameday="2026-11-22", home_team="CHI", away_team="NO"),
    # week 12: KC at BUF on Thanksgiving
    dict(week=12, game_type="REG", gameday="2026-11-26", home_team="BUF", away_team="KC"),
    dict(week=12, game_type="REG", gameday="2026-11-29", home_team="SF", away_team="SEA"),
    dict(week=12, game_type="REG", gameday="2026-11-29", home_team="CIN", away_team="NO"),
])
TEAMS = {"Kenneth Walker III": "KC", "Devontez Walker": "BAL", "Jaxon Smith-Njigba": "SEA", "Chris Olave": "NO",
         "Omarion Hampton": "LAC"}
MINE = {"Kenneth Walker III", "Jaxon Smith-Njigba", "Chris Olave", "Omarion Hampton"}
THU = dt.date(2026, 11, 19)


def chk(text):
    return check_text(text, TEAMS, SCHED, THU, prefer=MINE)


def test_current_week_is_the_week_in_progress():
    assert current_week(SCHED, THU) == 11
    assert current_week(SCHED, dt.date(2026, 11, 23)) == 12


def test_walker_tonight_is_blocked():
    bad = chk("Walker plays tonight with a questionable tag (groin).")
    assert len(bad) == 1 and "Sunday Nov 22 vs ARI" in bad[0]


def test_wrong_day_and_opponent_from_another_week_are_blocked():
    bad = chk("Walker plays Thursday @ LAR.")
    assert any("not Thursday" in b for b in bad) and any("not LA" in b for b in bad)


def test_bye_claims():
    assert any("on bye in week 11" in b for b in chk("Smith-Njigba plays Sunday."))
    assert any("not on bye" in b for b in chk("Olave is on bye this week."))
    assert chk("Smith-Njigba is on bye in week 11.") == []


def test_correct_advice_passes():
    text = ("Walker doesn't play tonight. KC hosts Arizona on Sunday in week 11. His Thursday game is next week, "
            "Thanksgiving at Buffalo. Olave plays at Chicago on Sunday. Hampton (LAC) is unchanged. "
            "Smith-Njigba plays at San Francisco in week 12.")
    assert chk(text) == [], chk(text)


def test_next_week_and_thanksgiving():
    assert chk("Walker plays Thursday next week at Buffalo.") == []
    assert any("Thanksgiving" in b for b in chk("Olave plays on Thanksgiving."))


def test_stale_questionable_tag_is_flagged():
    ros = pd.DataFrame([
        dict(Player="Bhayshul Tuten", Team="JAX", Status="Questionable", **{"Injury News": "Questionable, est. return Nov 22 (ankle)"}),
        dict(Player="Travis Etienne", Team="NO", Status="Questionable", **{"Injury News": "Questionable, est. return Nov 15 (hamstring)"}),
        dict(Player="Chris Olave", Team="NO", Status=None, **{"Injury News": None}),
    ])
    bad = stale_tags(ros, SCHED, 11)
    assert len(bad) == 1 and "Travis Etienne" in bad[0]


def test_negative_claims():
    assert chk("Walker doesn't play tonight.") == []
    assert any("DOES play" in b for b in chk("Walker does not play Sunday."))


def test_weekday_as_deadline_is_not_a_claim():
    assert chk("Smith-Njigba is on bye in week 11, so have the deal process after Sunday's games.") == []
    assert chk("Send the Olave offer by Monday.") == []


def test_team_named_claims_and_several_players():
    assert chk("The Hampton + Lemon for Olave deal is fine, since New Orleans plays at Chicago on Sunday.") == []
    assert chk("KC hosts Arizona on Sunday in week 11.") == []
    assert any("KC" in b for b in chk("KC plays Thursday in week 11."))
    assert any("SEA" in b for b in chk("Seattle plays Sunday."))
