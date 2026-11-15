"""Do WRs (and RB / TE) who have just become depth-chart starters beat the model's projection?
    PYTHONPATH=src python3 scripts/study_new_starter.py burbs
New starter at cut-off week T: listed as a starter on the depth chart for week T+1, but a starter in at most
a third of weeks 1..T. Residual = actual PPG after T minus the model's projection (current fit).
Depth charts: 2021-24 weekly (depth_team 1), 2025 daily ESPN snapshots (WR pos_rank <= 3, RB / TE rank 1)."""
import sys, json, datetime as dt
import numpy as np, polars as pl
from ffmodel import data
from ffmodel.config import DATA
lg = sys.argv[1] if len(sys.argv) > 1 else "burbs"
f = json.load(open(DATA / "processed" / "ros_fit.json"))
NST = {"WR": 3, "RB": 1, "TE": 1}

def starters(season):
    """season, week, player_id, position for depth-chart starters going into each week."""
    d = data.depth_charts([season])[0]
    if "dt" not in d.columns:
        d = d.filter(pl.col("game_type") == "REG", pl.col("depth_team") == "1", pl.col("depth_position").is_in(["WR", "RB", "TE"]),
                     pl.col("gsis_id").is_not_null())
        return d.select(pl.col("week").cast(pl.Int32), pl.col("gsis_id").alias("player_id"), pl.col("depth_position").alias("position")).unique()
    sch = data.schedules([season]).filter(pl.col("game_type") == "REG").group_by("week").agg(pl.col("gameday").min())
    d = d.filter(pl.col("pos_abb").is_in(["WR", "RB", "TE"]), pl.col("gsis_id").is_not_null()).with_columns(pl.col("dt").str.slice(0, 10).alias("day"))
    days = sorted(d["day"].unique().to_list())
    out = []
    for w, gd in sch.sort("week").iter_rows():
        cut = (dt.date.fromisoformat(gd) - dt.timedelta(days=2)).isoformat()
        prev = [x for x in days if x <= cut]
        if not prev:
            continue
        s = d.filter(pl.col("day") == prev[-1])
        s = s.filter(pl.col("dt") == s["dt"].max())
        s = s.filter(pl.col("pos_rank") <= pl.col("pos_abb").replace_strict(NST, return_dtype=pl.Int32))
        out.append(s.select(pl.lit(int(w)).cast(pl.Int32).alias("week"), pl.col("gsis_id").alias("player_id"), pl.col("pos_abb").alias("position")).unique())
    return pl.concat(out)

rows = []
for season in range(2021, 2026):
    st = starters(season)
    for T in (8, 9, 10):
        h = pl.read_parquet(DATA / "processed" / f"ros_history_wk{T}.parquet").filter(pl.col("season") == season)
        now = st.filter(pl.col("week") == T + 1).select("player_id").unique().with_columns(pl.lit(True).alias("st_now"))
        before = st.filter(pl.col("week") <= T).group_by("player_id").agg((pl.col("week").n_unique() / T).alias("st_share"))
        h = h.join(now, on="player_id", how="left").join(before, on="player_id", how="left").with_columns(
            pl.col("st_now").fill_null(False), pl.col("st_share").fill_null(0.0), pl.lit(T).alias("T"))
        rows.append(h)
h = pl.concat(rows, how="diagonal_relaxed")
print("pos  group                      n    model MAE  bias(model-actual)  | +bump MAE at best bump")
for p in ("WR", "RB", "TE"):
    d = h.filter(pl.col("position") == p, pl.col("n_after") >= 3, pl.col(f"pred_ppg_{lg}").is_not_null(), pl.col("status").fill_null("active") == "active")
    k, u, r, s = f["k"][lg][p], f["u"][lg][p], f["r"][lg][p], f["s"][lg][p]
    pri, n = d[f"pred_ppg_{lg}"].to_numpy(), d["n"].to_numpy()
    so = d[f"ppg_{lg}_sofar"].fill_null(0).to_numpy()
    obs = so + u * d["luck_pg"].fill_null(0).to_numpy() + r * d["role_trend"].fill_null(0).to_numpy() - s * d[f"sos_faced_{lg}"].fill_null(0).to_numpy()
    pred = (k * pri + n * obs) / (k + n) + s * d[f"sos_left_{lg}"].fill_null(0).to_numpy()
    y = d[f"ppg_{lg}_after"].to_numpy()
    stn, sh, se = d["st_now"].to_numpy(), d["st_share"].to_numpy(), d["season"].to_numpy()
    groups = {"new starter (<=1/3 before)": stn & (sh <= 1 / 3), "new starter, model < 8": stn & (sh <= 1 / 3) & (pred < 8),
              "established starter": stn & (sh > 2 / 3), "lost the job (was, not now)": ~stn & (sh > 2 / 3), "never a starter": ~stn & (sh <= 1 / 3)}
    for g, m in groups.items():
        if m.sum() < 8:
            print(f"{p:4} {g:26} {int(m.sum()):4}"); continue
        res = y[m] - pred[m]
        best = min(np.arange(-4, 6.01, 0.25), key=lambda b: np.mean(np.abs(res - b)))
        print(f"{p:4} {g:26} {int(m.sum()):4}   {np.mean(np.abs(res)):.2f}      {-res.mean():+.2f}              | {np.mean(np.abs(res - best)):.2f} at {best:+.2f}")
    m = groups["new starter (<=1/3 before)"]
    print("     new starters by season (n, model-actual):", {int(x): (int((m & (se == x)).sum()), round(float(-(y - pred)[m & (se == x)].mean()), 2)) for x in sorted(set(se)) if (m & (se == x)).sum() > 0})
    # leave-one-season-out: bump fitted on other seasons, applied to the held-out season
    tot_a = tot_b = cnt = 0
    for x in sorted(set(se)):
        tr, te = m & (se != x), m & (se == x)
        if te.sum() == 0 or tr.sum() < 8: continue
        b = np.median((y - pred)[tr])
        tot_a += np.abs((y - pred)[te]).sum(); tot_b += np.abs((y - pred)[te] - b).sum(); cnt += te.sum()
    if cnt: print(f"     held-out seasons: MAE {tot_a / cnt:.2f} -> {tot_b / cnt:.2f} with the bump fitted on the other seasons (n={cnt})")

print("\nFLOOR TEST (WR): projection = max(model, floor) for new depth-chart starters")
p = "WR"
d = h.filter(pl.col("position") == p, pl.col("n_after") >= 3, pl.col(f"pred_ppg_{lg}").is_not_null(), pl.col("status").fill_null("active") == "active")
k, u, r, s = f["k"][lg][p], f["u"][lg][p], f["r"][lg][p], f["s"][lg][p]
pri, n = d[f"pred_ppg_{lg}"].to_numpy(), d["n"].to_numpy()
obs = d[f"ppg_{lg}_sofar"].fill_null(0).to_numpy() + u * d["luck_pg"].fill_null(0).to_numpy() + r * d["role_trend"].fill_null(0).to_numpy() - s * d[f"sos_faced_{lg}"].fill_null(0).to_numpy()
pred = (k * pri + n * obs) / (k + n) + s * d[f"sos_left_{lg}"].fill_null(0).to_numpy()
y = d[f"ppg_{lg}_after"].to_numpy(); stn, sh, se = d["st_now"].to_numpy(), d["st_share"].to_numpy(), d["season"].to_numpy()
m = stn & (sh <= 1 / 3)
print("established starters, actual PPG after: quartiles", np.round(np.percentile(y[stn & (sh > 2 / 3)], [10, 25, 50, 75]), 1))
print("new starters: actual after quartiles", np.round(np.percentile(y[m], [10, 25, 50, 75]), 1), "model", np.round(np.percentile(pred[m], [10, 25, 50, 75]), 1))
for fl in (0, 5, 6, 7, 8, 9):
    pp = np.maximum(pred, fl)
    per = {int(x): round(float(np.mean(np.abs(y - pp)[m & (se == x)])), 2) for x in sorted(set(se))}
    print(f"floor {fl}: MAE {np.mean(np.abs(y - pp)[m]):.2f}  bias {np.mean((pp - y)[m]):+.2f}  moved {int((pred[m] < fl).sum())} of {int(m.sum())}  by season {per}")
m2 = ~stn & (sh > 2 / 3)
print("\nLOST THE JOB (WR): projection x factor")
for fa in (1.0, 0.9, 0.8, 0.7, 0.6):
    print(f"x{fa}: MAE {np.mean(np.abs(y - pred * fa)[m2]):.2f} bias {np.mean((pred * fa - y)[m2]):+.2f}", {int(x): round(float(np.mean(np.abs(y - pred * fa)[m2 & (se == x)])), 2) for x in sorted(set(se)) if (m2 & (se == x)).sum()})
