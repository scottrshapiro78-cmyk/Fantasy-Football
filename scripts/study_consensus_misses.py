"""Nov 26 2026: why the weekly model makes confident calls against every expert and loses (Andrews, Dak vs Lawrence).

Step 1 (this file, `build`): one row per player-week, 2021-2026, weeks 5-17, with the model's own pieces as of the
Tuesday before the game (no expert input):
  base  = ROS formula at thru = t-1 without the remaining-schedule term (what build_weekly calls Base PPG)
  eff   = this opponent's points allowed to the position vs expectation (opponent.defense_effects, games <= t-1)
  imp   = team's implied points this week (spread + total); avg_imp = its average this season (weeks <= t, as production)
plus recent-form fields (last 3 / last game points, usage last 3) and the FantasyPros weekly consensus rank (ECR),
which is used ONLY to grade disagreements, never as a model input.
Step 2 (`grade`): production weights vs variants, leave-one-season-out, and the disagreement table.

PYTHONPATH=src python3 scripts/study_consensus_misses.py build
PYTHONPATH=src python3 scripts/study_consensus_misses.py grade
"""
import json
import sys

import numpy as np
import polars as pl

from ffmodel import data, ros
from ffmodel.board import projections
from ffmodel.config import DATA
from ffmodel.opponent import defense_effects

OUT = DATA / "processed" / "consensus_study.parquet"
SKILL = ["QB", "RB", "WR", "TE"]
LG = ("burbs", "cbs")


def implied() -> pl.DataFrame:
    sc = data.schedules(list(range(2021, 2027))).filter(pl.col("game_type") == "REG").select(
        "season", pl.col("week").cast(pl.Int32), "home_team", "away_team", "spread_line", "total_line", "gameday")
    hi, ai = (pl.col("total_line") + pl.col("spread_line")) / 2, (pl.col("total_line") - pl.col("spread_line")) / 2
    g = pl.concat([sc.select("season", "week", "gameday", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"),
                             hi.alias("imp"), ai.alias("opp_imp"), pl.lit(1).alias("home")),
                   sc.select("season", "week", "gameday", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"),
                             ai.alias("imp"), hi.alias("opp_imp"), pl.lit(0).alias("home"))]).sort("season", "week")
    # production: team's average implied points over weeks <= this week (incl. this one)
    g = g.with_columns((pl.col("imp").cum_sum().over("season", "team") / pl.col("imp").cum_count().over("season", "team")).alias("avg_imp"),
                       (pl.col("opp_imp").cum_sum().over("season", "team") / pl.col("opp_imp").cum_count().over("season", "team")).alias("avg_oimp"))
    return g


def ecr_weekly() -> pl.DataFrame:
    ecr = pl.read_parquet(DATA / "external" / "db_fpecr.parquet").filter(
        pl.col("pos").is_in(SKILL), pl.col("page_type").str.starts_with("weekly-"), pl.col("ecr_type") == "wp")
    ids = (pl.read_parquet(DATA / "processed" / "player_seasons.parquet").sort("season")
           .select("player_id", "yahoo_id", "merge_name", "position").unique(["player_id"], keep="last"))
    yid = ids.drop_nulls("yahoo_id").unique("yahoo_id").select("yahoo_id", pl.col("player_id").alias("pid_y"))
    nm = ids.group_by("merge_name", "position").agg(pl.col("player_id").first().alias("pid_n"), pl.len().alias("k")).filter(pl.col("k") == 1)
    ecr = (ecr.join(yid, on="yahoo_id", how="left")
           .with_columns(pl.col("mergename").str.to_lowercase().str.replace_all(r"[.'’]", "").str.replace_all(r"\s+(jr|sr|ii|iii|iv)$", "").alias("mn"))
           .join(nm, left_on=["mn", "pos"], right_on=["merge_name", "position"], how="left")
           .with_columns(pl.coalesce("pid_y", "pid_n").alias("player_id"), pl.col("scrape_date").str.to_date()).drop_nulls("player_id"))
    sun = (data.schedules(list(range(2021, 2027))).filter(pl.col("game_type") == "REG")
           .group_by("season", pl.col("week").cast(pl.Int32)).agg(pl.col("gameday").str.to_date().max().alias("sun")))
    rows = []
    for s, w, d in sun.iter_rows():
        e = ecr.filter(pl.col("scrape_date") <= d, pl.col("scrape_date") >= d - pl.duration(days=6))
        if e.is_empty():
            continue
        e = e.filter(pl.col("scrape_date") == e["scrape_date"].max()).sort("ecr").unique("player_id", keep="first")
        rows.append(e.select("player_id", pl.col("ecr").cast(pl.Float64), pl.lit(s).alias("season"), pl.lit(w).cast(pl.Int32).alias("week")))
    return pl.concat(rows)


def build() -> None:
    fit = json.loads((DATA / "processed" / "ros_fit.json").read_text())
    pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").with_columns(pl.col("week").cast(pl.Int32))
    imp = implied()
    ecr = ecr_weekly()
    parts = []
    for s in range(2021, 2027):
        last_wk = 11 if s == 2026 else 17
        pri = {lg: projections(s, lg).select("player_id", pl.col(f"pred_ppg_{lg}").alias(f"prior_{lg}"), "rookie") for lg in LG}
        ws = pw.filter(pl.col("season") == s)
        for t in range(5, last_wk + 1):
            thru = t - 1
            so = ros.to_date(s, thru)
            hist = ws.filter(pl.col("week") <= thru).sort("week")
            form = hist.group_by("player_id").agg(
                *[pl.col(f"fpts_{lg}").tail(3).mean().alias(f"l3_{lg}") for lg in LG],
                *[pl.col(f"fpts_{lg}").last().alias(f"l1_{lg}") for lg in LG],
                pl.col("attempts").tail(3).mean().alias("att_l3"), pl.col("attempts").mean().alias("att_pg"))
            y = ws.filter(pl.col("week") == t).select("player_id", "position", "team", "player_display_name",
                                                       *[pl.col(f"fpts_{lg}").alias(f"y_{lg}") for lg in LG])
            d = y.join(so.select("player_id", "n", "luck_pg", "role_trend", "xfp_pg", "xfp_last3",
                                 *[f"ppg_{lg}_sofar" for lg in LG]), on="player_id", how="inner").join(form, on="player_id", how="left")
            for lg in LG:
                d = d.join(pri[lg].select("player_id", f"prior_{lg}"), on="player_id", how="left")
                ss = ros.sos(s, thru, lg).select("player_id", pl.col("sos_faced").alias(f"sf_{lg}"))
                eff = defense_effects(hist, f"fpts_{lg}").select("position", pl.col("opponent_team").alias("opp"),
                                                                 pl.col(f"def_fpts_{lg}").alias(f"eff_{lg}"))
                d = d.join(ss, on="player_id", how="left")
                d = d.join(imp.filter(pl.col("season") == s, pl.col("week") == t).select("team", "opp"), on="team", how="left") if "opp" not in d.columns else d
                d = d.join(eff, on=["position", "opp"], how="left")
            d = d.join(pri["burbs"].select("player_id", "rookie"), on="player_id", how="left")
            d = d.join(imp.filter(pl.col("season") == s, pl.col("week") == t).drop("season", "week", "opp"), on="team", how="left")
            d = d.join(ecr.filter(pl.col("season") == s, pl.col("week") == t).drop("season", "week"), on="player_id", how="left")
            parts.append(d.with_columns(pl.lit(s).alias("season"), pl.lit(t).cast(pl.Int32).alias("week")))
        print("built", s, flush=True)
    df = pl.concat(parts, how="diagonal_relaxed").filter(pl.col("position").is_in(SKILL))
    # league-wide average implied points this season through this week (for the level term)
    df.write_parquet(OUT)
    (DATA / "processed" / "consensus_study_fit.json").write_text(json.dumps(fit))
    print(OUT, df.height)


def injury_features() -> pl.DataFrame:
    """Pre-game facts for week t (no expert input): Questionable / last practice on that week's report, and whether he
    missed his team's last game and is back. Also fresh absences of same-position teammates (tested, see notes)."""
    yrs = list(range(2021, 2027))
    pw = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(
        pl.col("season") >= 2021, pl.col("position").is_in(SKILL)).with_columns(pl.col("week").cast(pl.Int32))
    inj = (data.injuries(yrs).filter(pl.col("game_type") == "REG")
           .select("season", pl.col("week").cast(pl.Int32), pl.col("gsis_id").alias("player_id"), "report_status", "practice_status")
           .unique(["season", "week", "player_id"]))
    rw = (data.rosters_weekly(yrs).filter(pl.col("game_type") == "REG")
          .select("season", pl.col("week").cast(pl.Int32), pl.col("gsis_id").alias("player_id"), pl.col("status").alias("rstat"),
                  pl.col("team").alias("rteam")).unique(["season", "week", "player_id"]))
    d = pl.read_parquet(OUT)
    rows = []
    for (s, t), g in d.group_by(["season", "week"]):
        h = pw.filter(pl.col("season") == s, pl.col("week") < t)
        me = h.group_by("player_id").agg(pl.col("week").max().alias("lastwk"), pl.len().alias("g"),
                                         pl.col("fpts_burbs").mean().alias("ppg_b"), pl.col("position").last(), pl.col("team").last())
        tl = h.group_by("team").agg(pl.col("week").max().alias("tl"))
        it, rt = inj.filter(pl.col("season") == s, pl.col("week") == t), rw.filter(pl.col("season") == s, pl.col("week") == t)
        st = (me.join(it, on="player_id", how="left").join(rt, on="player_id", how="left")
              .with_columns(pl.coalesce("rteam", "team").alias("tm")).join(tl.rename({"team": "tm"}), on="tm", how="left"))
        out_ = (pl.col("report_status").is_in(["Out", "Doubtful"]).fill_null(False)
                | pl.col("rstat").is_in(["RES", "INA", "CUT", "SUS", "PUP"]).fill_null(False))
        st = st.with_columns((out_ & (pl.col("lastwk") >= pl.col("tl") - 1)).alias("isout"))
        tp = st.filter(pl.col("g") >= 2).group_by("tm", "position").agg((pl.col("ppg_b") * pl.col("isout")).sum().alias("outppg"))
        x = (g.select("player_id", "season", "week", "team", "position")
             .join(me.select("player_id", "lastwk", "g"), on="player_id", how="left")
             .join(tl, on="team", how="left").join(it.select("player_id", "report_status", "practice_status"), on="player_id", how="left")
             .join(tp.rename({"tm": "team"}), on=["team", "position"], how="left"))
        rows.append(x)
    return pl.concat(rows, how="diagonal_relaxed").select(
        "player_id", "season", "week", pl.col("outppg").fill_null(0),
        (pl.col("report_status") == "Questionable").fill_null(False).cast(pl.Int8).alias("q"),
        pl.col("practice_status").str.contains("Did Not").fill_null(False).cast(pl.Int8).alias("dnp"),
        ((pl.col("g") >= 3) & (pl.col("lastwk") < pl.col("tl"))).fill_null(False).cast(pl.Int8).alias("returning"))


MATCH = {"QB": 1.0, "RB": 0.45, "WR": 0.35, "TE": 0.55}  # production before Nov 26 2026
VEGAS = {"QB": 0.0, "RB": 0.075, "WR": 0.0, "TE": 0.075}
TOP = {"QB": 20, "RB": 40, "WR": 50, "TE": 18}
FEATS = ["base", "eff", "dimp", "implvl", "q", "returning"]  # QB; RB/WR/TE add dnp
VARIANTS = {"vegas+shrink": ["base", "eff", "dimp", "implvl"], "+recent form": ["base", "eff", "dimp", "implvl", "l3dev"],
            "+prior gap": ["base", "eff", "dimp", "implvl", "pri_gap"], "+injury tags": None, "+teammate outs": None}


def frame(lg: str = "burbs"):
    import pandas as pd
    fit = json.loads((DATA / "processed" / "ros_fit.json").read_text())
    d = pl.read_parquet(OUT).join(injury_features(), on=["player_id", "season", "week"], how="left").to_pandas()
    low = d.groupby("position")[f"prior_{lg}"].quantile(0.2)
    d["prior"] = d[f"prior_{lg}"].fillna(d.position.map(low))
    d["y"], d["sofar"], d["eff"], d["sf"] = d[f"y_{lg}"], d[f"ppg_{lg}_sofar"], d[f"eff_{lg}"].fillna(0), d[f"sf_{lg}"].fillna(0)
    k, u = d.position.map(fit["k"][lg]), d.position.map(fit["u"][lg])
    r, s_ = d.position.map(fit["r"][lg]), d.position.map(fit["s"][lg])
    rk = d.rookie.fillna(False).astype(bool)
    k = np.where(rk, k * d.position.map(ros.ROOKIE_K_MULT).fillna(1.0), k)
    r = np.where(rk, np.maximum(r, d.position.map(ros.ROOKIE_R).fillna(0.0)), r)
    d["base"] = (k * d.prior + d.n * (d.sofar + u * d.luck_pg + r * d.role_trend - s_ * d.sf)) / (k + d.n)
    d["dimp"] = d.imp - d.avg_imp
    d["implvl"] = d.imp - d.season.map(d.groupby("season").imp.mean())
    d["l3dev"] = (d[f"l3_{lg}"] - d.sofar).fillna(0)
    d["pri_gap"] = d.prior - d.sofar
    d["prod"] = d.base + d.position.map(MATCH) * d.eff + d.position.map(VEGAS) * d.dimp
    grp = ["season", "week", "position"]
    d["mr"], d["er"] = d.groupby(grp)["prod"].rank(ascending=False), d.groupby(grp)["ecr"].rank()
    return d[(d.mr <= d.position.map(TOP)) | (d.er <= d.position.map(TOP))].copy()


def loso(q, feats):
    import pandas as pd
    import statsmodels.api as sm
    pred, coefs = pd.Series(index=q.index, dtype=float), []
    for s in sorted(q.season.unique()):
        tr, te = q[q.season != s], q[q.season == s]
        m = sm.QuantReg(tr.y, sm.add_constant(tr[feats])).fit(q=0.5, max_iter=5000)
        coefs.append(m.params)
        pred[te.index] = m.predict(sm.add_constant(te[feats], has_constant="add"))
    return pred, pd.concat(coefs, axis=1).median(axis=1)


def disagreements(q, col):
    """Same-position pairs in a week where the model's order and the FantasyPros order differ: share the model got right,
    all of them and those where the model's margin was 2+ points."""
    import pandas as pd
    out = []
    for _, g in q[q.ecr.notna()].groupby(["season", "week", "position"]):
        m, e, y = g[col].values, g.ecr.values, g.y.values
        i, j = np.triu_indices(len(g), 1)
        ok = (y[i] != y[j]) & (np.sign(m[i] - m[j]) != np.sign(e[j] - e[i]))
        out.append(pd.DataFrame({"gap": np.abs(m[i] - m[j])[ok], "right": (np.sign(m[i] - m[j]) == np.sign(y[i] - y[j]))[ok]}))
    x = pd.concat(out)
    return len(x), x.right.mean(), int((x.gap >= 2).sum()), x[x.gap >= 2].right.mean()


def grade() -> None:
    import pandas as pd
    q = frame()
    rows = []
    for p in SKILL:
        g = q[q.position == p].copy()
        feats = FEATS + (["dnp"] if p != "QB" else [])
        specs = {**{k: v for k, v in VARIANTS.items() if v}, "+injury tags": feats, "+teammate outs": feats + ["outppg"]}
        m0 = (g["prod"] - g.y).abs()
        rows.append((p, "production", m0.mean(), 0.0, "", *disagreements(g, "prod"), *disagreements(g[g.season == 2026], "prod")))
        for name, f in specs.items():
            g[name], c = loso(g, f)
            m1 = (g[name] - g.y).abs()
            wins = sum(m1[g.season == s].mean() < m0[g.season == s].mean() for s in g.season.unique())
            rows.append((p, name, m1.mean(), 100 * (m1.mean() / m0.mean() - 1), f"{wins}/6", *disagreements(g, name),
                         *disagreements(g[g.season == 2026], name)))
            if name == "+injury tags":
                print(p, "coefficients (median over folds):", c.round(3).to_dict())
    t = pd.DataFrame(rows, columns=["pos", "variant", "miss", "miss_vs_prod_%", "seasons_better", "disagree_n", "model_right",
                                    "disagree_2plus_n", "model_right_2plus", "n_2026", "right_2026", "n2_2026", "right2_2026"])
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        grade()
