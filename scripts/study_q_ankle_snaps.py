"""When a TE/WR listed Questionable plays, does he play a full snap load or a reduced role?  2018-2025.
PYTHONPATH=src python3 scripts/study_q_ankle_snaps.py
Row = WR/TE on the final report as Questionable who played. Compared with HIS OWN average over his previous 4 games played
(snap %, targets, fantasy points), only players with a real role (4-game snap % >= 50%).  Groups: ankle / other leg / non-leg;
practice Limited vs DNP vs Full."""
import numpy as np, pandas as pd, polars as pl
from ffmodel import data
from ffmodel.config import DATA

SS = list(range(2018, 2026))
ids = data.player_ids().select("gsis_id", "pfr_id").drop_nulls().to_pandas()
sn = data.snap_counts(SS).filter(pl.col("game_type") == "REG").select("season", "week", "pfr_player_id", "offense_pct").to_pandas()
sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id").rename(columns={"gsis_id": "player_id"})
pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season").is_in(SS), pl.col("position").is_in(["WR", "TE"]), pl.col("week") <= 18).select("player_id", "position", "season", "week", "targets", "fpts_burbs").to_pandas()
g = pw.merge(sn[["player_id", "season", "week", "offense_pct"]], on=["player_id", "season", "week"], how="inner").sort_values(["player_id", "season", "week"])
for c in ["offense_pct", "targets", "fpts_burbs"]:
    g[f"b_{c}"] = g.groupby(["player_id", "season"])[c].transform(lambda s: s.shift().rolling(4, min_periods=3).mean())
inj = data.injuries(SS).filter(pl.col("game_type") == "REG", pl.col("report_status") == "Questionable").to_pandas().drop_duplicates(["season", "week", "gsis_id"], keep="last")
LEG = ("ankle", "knee", "foot", "hamstring", "calf", "groin", "quad", "thigh", "toe", "achilles", "hip", "shin")
inj["body"] = inj.report_primary_injury.fillna("").str.lower()
inj["grp"] = np.where(inj.body.str.contains("ankle"), "ankle", np.where(inj.body.str.contains("|".join(LEG)), "other leg", "non-leg"))
inj["pr"] = inj.practice_status.fillna("").map(lambda s: "DNP" if s.startswith("Did Not") else "Lim" if s.startswith("Limited") else "Full" if s.startswith("Full") else "?")
d = g.merge(inj[["season", "week", "gsis_id", "grp", "pr"]].rename(columns={"gsis_id": "player_id"}), on=["player_id", "season", "week"])
d = d[d.b_offense_pct >= 0.5].copy()
d["snap_ratio"] = d.offense_pct / d.b_offense_pct
d["tgt_diff"] = d.targets.fillna(0) - d.b_targets
d["pts_diff"] = d.fpts_burbs - d.b_fpts_burbs
d["snap_drop_20"] = d.snap_ratio < 0.8
# healthy control: same players, not on the report
ctl = g[(g.b_offense_pct >= 0.5)].merge(inj[["season", "week", "gsis_id"]].rename(columns={"gsis_id": "player_id"}), on=["player_id", "season", "week"], how="left", indicator=True)
ctl = ctl[ctl._merge == "left_only"]
ctl = ctl.assign(snap_ratio=ctl.offense_pct / ctl.b_offense_pct, tgt_diff=ctl.targets.fillna(0) - ctl.b_targets, pts_diff=ctl.fpts_burbs - ctl.b_fpts_burbs, snap_drop_20=lambda x: x.snap_ratio < 0.8, grp="not listed", pr="-")
agg = dict(n=("snap_ratio", "size"), snap_vs_norm=("snap_ratio", "median"), share_big_snap_drop=("snap_drop_20", "mean"),
           targets_vs_norm=("tgt_diff", "mean"), pts_vs_norm=("pts_diff", "mean"), base_pts=("b_fpts_burbs", "mean"))
for pos in ["TE", "WR"]:
    a = pd.concat([d[d.position == pos], ctl[ctl.position == pos]])
    print(f"\n{pos}: Questionable and played, vs his own last-4 average")
    print(a.groupby("grp").agg(**agg).round(2).to_string())
    print(a[a.grp == "ankle"].groupby("pr").agg(**agg).round(2).to_string())
