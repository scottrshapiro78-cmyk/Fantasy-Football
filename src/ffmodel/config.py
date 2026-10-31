from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_leagues() -> dict:
    with open(ROOT / "config" / "leagues.yaml") as f:
        return yaml.safe_load(f)


def league(key: str) -> dict:
    return load_leagues()[key]
