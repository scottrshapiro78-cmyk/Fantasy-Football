"""Weekly scorecard: model-only vs blended vs FantasyPros consensus vs Footballguys / RotoBaller / Yahoo, on actual points.
PYTHONPATH=src python3 scripts/expert_scorecard.py <week>
Set = players in the frozen weekly tracking snapshot who PLAYED (have a stat row) and whose pre-play projection >= 5.
Each outside source is scored on the players it covers, with the model on that same set (head-to-head).
Projections are compared before the play-chance discount (all sources project assuming he plays), i.e. proj/play.
Appends to /mnt/project-files/ff-model/expert_scorecard.csv."""
import re, sys, glob
import numpy as np, pandas as pd, polars as pl
from ffmodel.config import DATA

WK = int(sys.argv[1]); EX = "/mnt/project-files/ff-model/inputs/expert/"
norm = lambda s: re.sub(r"[^a-z]", "", re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower()))
act = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == 2026, pl.col("week") == WK).to_pandas()
rows = []
for lg in ["burbs", "cbs"]:
    t = pd.read_csv(DATA / "tracking" / f"weekly_2026_wk{WK}_{lg}.csv")
    t = t[t.pos.isin(["QB", "RB", "WR", "TE"]) & (t.play > 0)].copy()
    t["blend"] = t.proj / (t.play / 100)
    t["model"] = t.blend - t.expert_adj.fillna(0)
    t = t.merge(act[["player_id", f"fpts_{lg}"]].rename(columns={f"fpts_{lg}": "act"}), on="player_id")
    t = t[t.model >= 5]; t["k"] = t.player.map(norm)
    src = {"FantasyPros": t.set_index("k").consensus}
    fbg = [pd.read_csv(f) for f in glob.glob(f"{EX}fbg_weekly_2026wk{WK}_*_{lg}.csv") if re.search(r"_(qb|rb|wr|te)_", f)]
    if fbg:
        f = pd.concat(fbg); src["Footballguys"] = f.assign(k=f.Player.map(norm)).drop_duplicates("k").set_index("k").Points
    if lg == "burbs":
        for f in glob.glob(f"{EX}rotoballer_weekly_ppr_2026wk{WK}.csv"):
            r = pd.read_csv(f); src["RotoBaller"] = r.assign(k=r.Player.map(norm)).drop_duplicates("k").set_index("k").PTS
        for f in glob.glob(f"{EX}yahoo_weekly_2026wk{WK}_burbs.csv"):
            y = pd.read_csv(f); y = y[y.yahoo_pts > 0]; src["Yahoo"] = y.assign(k=y.player.map(norm)).drop_duplicates("k").set_index("k").yahoo_pts
    base = dict(week=WK, league=lg)
    for pos in ["ALL", "QB", "RB", "WR", "TE"]:
        g = t if pos == "ALL" else t[t.pos == pos]
        rows.append({**base, "pos": pos, "source": "Model only", "vs": "all", "n": len(g), "mae": (g.act - g.model).abs().mean()})
        rows.append({**base, "pos": pos, "source": "Model + blend", "vs": "all", "n": len(g), "mae": (g.act - g.blend).abs().mean()})
        for name, s in src.items():
            h = g.assign(x=g.k.map(s)).dropna(subset=["x"])
            rows.append({**base, "pos": pos, "source": name, "vs": "head-to-head", "n": len(h), "mae": (h.act - h.x).abs().mean(),
                         "model_mae_same_set": (h.act - h.model).abs().mean(), "blend_mae_same_set": (h.act - h.blend).abs().mean()})
    # Edge flags (model-only vs consensus)
    gap = t.model - t.consensus
    for flag, m in [("Fade", gap <= -2), ("Model high", gap >= 2)]:
        e = t[m]
        right = ((e.act - e.model).abs() < (e.act - e.consensus).abs()).mean() if len(e) else np.nan
        rows.append({**base, "pos": "ALL", "source": f"Edge: {flag}", "vs": "model closer than consensus", "n": len(e), "mae": right})
out = pd.DataFrame(rows).round(3)
path = "/mnt/project-files/ff-model/expert_scorecard.csv"
try:
    old = pd.read_csv(path); old = old[old.week != WK]; out = pd.concat([old, out])
except FileNotFoundError:
    pass
out.to_csv(path, index=False)
print(out[out.week == WK].to_string(index=False))
