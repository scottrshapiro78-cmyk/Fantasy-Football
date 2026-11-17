import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_rules as c

POS = {"Lawrence": "QB", "Dak": "QB", "W": "RB", "T": "RB", "Wa": "RB", "J": "RB", "McG": "RB", "L": "WR", "M": "WR",
       "Ty": "WR", "O": "WR", "B": "TE", "Bass": "K", "MIN": "DEF", "Ges": "TE"}


def test_plan_cannot_drop_only_kicker():
    assert any("no K" in x for x in c.check_plan(list(POS), ["Ges"], ["Bass"], POS))


def test_plan_ok_when_dropping_spare_rb():
    assert c.check_plan([p for p in POS if p != "Ges"], ["Ges"], ["McG"], POS) == []


def test_keep_list_loaded():
    assert {"Rico Dowdle", "Emmett Johnson"} <= c.keep_list()
