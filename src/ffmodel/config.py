from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
import os

# FF_DATA: an alternate data folder (e.g. data_2006 for long backtests) so production files are untouched
DATA = Path(os.environ["FF_DATA"]).resolve() if os.environ.get("FF_DATA") else ROOT / "data"


def load_leagues() -> dict:
    with open(ROOT / "config" / "leagues.yaml") as f:
        return yaml.safe_load(f)


def league(key: str) -> dict:
    return load_leagues()[key]
