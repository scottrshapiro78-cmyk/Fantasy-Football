"""Weekly rankings (start/sit) for one week, both leagues.

PYTHONPATH=src python3 scripts/build_weekly.py [season=2026] [week=9]
Writes data/output/weekly_<season>_wk<week>_<league>.xlsx: Overall, QB, RB, WR, TE, K, DEF, FLEX, Notes, Column Guide.

Week projection = ROS PPG with the rest-of-season schedule taken out (talent + role + this season so far)
  + MATCH[pos] x this opponent's points allowed to the position vs expectation (this season)
  + VEGAS[pos] x (team's implied points this week - its average implied points this season)
then x chance he plays (injury report / roster status). Weights fitted on 2021-2025 weeks 5-17
(scratch backtest): matchups and lines move weekly scoring a little; defenses the most.
"""
import sys

import polars as pl
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

from ffmodel import data, ros
from ffmodel.board import _tiers
from ffmodel.config import DATA, load_leagues
from ffmodel.guide import GUIDE, add_guide
from ffmodel.opponent import defense_effects
from ffmodel.ros import dst_weeks

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
week = int(sys.argv[2]) if len(sys.argv) > 2 else 9
thru = week - 1
POS = ["QB", "RB", "WR", "TE", "K", "DEF"]
MATCH = {"QB": 1.0, "RB": 0.45, "WR": 0.35, "TE": 0.55, "K": 0.0, "DEF": 0.5}
VEGAS = {"QB": 0.0, "RB": 0.075, "WR": 0.0, "TE": 0.075, "K": 0.13, "DEF": -0.25}  # DEF uses the opponent's implied pts
QUESTIONABLE_PLAY = 0.85
DNP_PLAY = 0.65  # questionable and did not practice Wednesday (rough figure, not fitted)
CONCUSSION_PLAY = 0.55  # in protocol midweek: roughly half clear it in time (rough figure, not fitted)
TIER_SHAPE = {"QB": (24, 5), "RB": (40, 7), "WR": (50, 8), "TE": (20, 5), "K": (16, 3), "DEF": (16, 3)}

TEAMS = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

sch = data.schedules([season]).filter(pl.col("game_type") == "REG")
wk = sch.filter(pl.col("week") == week)
side = lambda t, o, home, imp, oimp: wk.select(pl.col(t).alias("team"), pl.col(o).alias("opp"), pl.lit(home).alias("home"),
                                             "gameday", "gametime", imp.alias("imp"), oimp.alias("opp_imp"))
hi, ai = (pl.col("total_line") + pl.col("spread_line")) / 2, (pl.col("total_line") - pl.col("spread_line")) / 2
games = pl.concat([side("home_team", "away_team", True, hi, ai), side("away_team", "home_team", False, ai, hi)])
allimp = pl.concat([sch.select("week", pl.col("home_team").alias("team"), hi.alias("imp")),
                    sch.select("week", pl.col("away_team").alias("team"), ai.alias("imp"))]).filter(pl.col("week") <= week)
avg_imp = allimp.group_by("team").agg(pl.col("imp").mean().alias("avg_imp"))
avg_oimp = pl.concat([sch.select("week", pl.col("home_team").alias("team"), ai.alias("o")),
                      sch.select("week", pl.col("away_team").alias("team"), hi.alias("o"))]).filter(
    pl.col("week") <= week).group_by("team").agg(pl.col("o").mean().alias("avg_oimp"))
games = games.join(avg_imp, on="team").join(avg_oimp, on="team")
byes = sorted(set(sch["home_team"]) - set(games["team"]))

wks = ros._weekly(season).filter(pl.col("week") <= thru)
last3 = (wks.sort("week").group_by("player_id")
            .agg(*[pl.col(f"fpts_{lg}").tail(3).mean().alias(f"l3_{lg}") for lg in ("burbs", "cbs")]))

GUIDE.update({
    "Wk Proj": "Projected fantasy points this week, after matchup, Vegas line and the chance he plays. Rank and start/sit on this.",
    "Opp": "This week's opponent ('@' = away).",
    "Kickoff": "Game day and time (ET).",
    "Base PPG": "His rest-of-season points per game before this week's matchup: preseason projection blended with this season's results and recent role.",
    "Matchup Adj": "Points added (+) or taken away (-) by this opponent: how that defense has done this season against his position (run D for RBs, pass D for QB/WR/TE). For a team defense: how many points that offense gives up to defenses.",
    "Team Implied Pts": "Points Vegas expects his team to score this week (from the over/under and spread). For a team defense: points the opponent is expected to score.",
    "Vegas Adj": "Points added or taken away because his team's implied total is higher or lower than its usual this season (fitted: matters for RB, TE, K and defenses; not for QB/WR once the matchup is counted).",
    "Play Chance": "Chance he plays this week: 0% if Out / Doubtful / IR, 85% if Questionable (65% if he missed Wednesday practice), 55% if in the concussion protocol, low for practice-squad or players with no games yet.",
    "Teammate Out Adj": "Points added because his team's top WR may sit this week (weighted by the chance he sits). When a team's top WR missed a game (2016-2025), the other WRs scored about 2 points more than usual and the top TE about 1 point more. If the top WR is ruled out, this becomes the full +2.0 / +0.9.",
    "Last 3 PPG": "Average points in his last 3 games.",
    "Pts Over Repl": "Week projection minus a replacement-level starter at his position (the last starter in a 12-team league). Ranks the Overall tab across positions.",
    "Status": "Injury report status (from the latest report you pasted) or roster status.",
})

out_dir = DATA / "output"
cfg = load_leagues()
for lg in ("burbs", "cbs"):
    col = f"pred_ppg_{lg}"
    d = ros.project(season, thru, lg)
    d = d.with_columns((pl.col(col) - pl.col("sched_adj")).alias("base"))
    # this week's matchup
    w = pl.read_parquet(DATA / "processed" / "player_weeks.parquet").filter(pl.col("season") == season, pl.col("week") <= thru)
    eff = defense_effects(w, f"fpts_{lg}").select("position", pl.col("opponent_team").alias("opp"), pl.col(f"def_fpts_{lg}").alias("eff"))
    dw = dst_weeks().filter(pl.col("season") == season, pl.col("week") <= thru)
    mu = dw[f"fpts_{lg}"].mean()
    off = dw.group_by("opponent_team").agg(((pl.col(f"fpts_{lg}") - mu).sum() / (pl.len() + 4)).alias("eff")).rename({"opponent_team": "opp"})
    eff = pl.concat([eff, off.with_columns(pl.lit("DEF").alias("position")).select("position", "opp", "eff")])
    d = (d.join(games, on="team", how="inner").join(eff, on=["position", "opp"], how="left")
           .join(last3.select("player_id", pl.col(f"l3_{lg}").alias("l3")), on="player_id", how="left"))
    m = pl.col("position").replace_strict(MATCH, return_dtype=pl.Float64)
    v = pl.col("position").replace_strict(VEGAS, return_dtype=pl.Float64)
    impd = pl.when(pl.col("position") == "DEF").then(pl.col("opp_imp") - pl.col("avg_oimp")).otherwise(pl.col("imp") - pl.col("avg_imp"))
    news = pl.col("news").fill_null("")
    play = (pl.when(pl.col("status").is_in(["out_week", "out_long", "season", "dated", "gone"])).then(0.0)
              .when(pl.col("status") == "practice").then(0.05)
              .when(pl.col("status") == "no_games").then(0.15)
              .when(news.str.to_lowercase().str.contains("concussion")).then(CONCUSSION_PLAY)
              .when(news.str.starts_with("Questionable") & news.str.contains("DNP")).then(DNP_PLAY)
              .when(news.str.starts_with("Questionable")).then(QUESTIONABLE_PLAY)
              .otherwise(1.0))
    d = d.with_columns((m * pl.col("eff").fill_null(0)).alias("match_adj"), (v * impd.fill_null(0)).alias("vegas_adj"),
                       play.alias("play"))
    # top WR (7+ targets a game) may sit: teammates pick up his work. 2016-2025, games the team's top WR
    # missed: other WRs scored about +2.0 pts and the top TE about +0.9 vs their usual (scratch study).
    # Scaled by the chance he sits.
    tg = (pl.read_parquet(DATA / "processed" / "player_weeks.parquet")
            .filter(pl.col("season") == season, pl.col("week") <= thru, pl.col("position") == "WR")
            .group_by("player_id").agg(pl.len().alias("g"), pl.col("targets").fill_null(0).mean().alias("tpg")))
    star = (d.join(tg, on="player_id").filter(pl.col("g") >= 3, pl.col("tpg") >= 7)
              .sort("tpg", descending=True).group_by("team").first()
              .select("team", pl.col("player_id").alias("star_id"), pl.col("player_display_name").alias("star"),
                      (1 - pl.col("play")).alias("sit")))
    d = d.join(star, on="team", how="left").with_columns(pl.col("sit").fill_null(0))
    te1 = pl.col("base").rank("ordinal", descending=True).over("team", "position") == 1
    bump = (pl.when(pl.col("player_id") == pl.col("star_id")).then(0.0)
              .when(pl.col("position") == "WR").then(2.0).when((pl.col("position") == "TE") & te1).then(0.9)
              .otherwise(0.0) * pl.col("sit"))
    d = d.with_columns(bump.alias("absence_adj"))
    d = d.with_columns(((pl.col("base") + pl.col("match_adj") + pl.col("vegas_adj") + pl.col("absence_adj")).clip(0)
                        * pl.col("play")).alias("wk"))
    # replacement = last starter in the league (flex filled by the best leftover RB/WR/TE)
    L = cfg[lg]
    n, ros_ = L["teams"], L["roster"]
    starters = {p: n * ros_.get(p, 0) for p in POS}
    rank = d.sort("wk", descending=True).with_columns(pl.int_range(1, pl.len() + 1).over("position").alias("pr"))
    left = rank.filter(pl.col("position").is_in(L["flex_eligible"]),
                       pl.col("pr") > pl.col("position").replace_strict(starters, return_dtype=pl.Int64)).head(n * ros_.get("FLEX", 0))
    for p in L["flex_eligible"]:
        starters[p] += left.filter(pl.col("position") == p).height
    repl = {p: (d.filter(pl.col("position") == p).sort("wk", descending=True)["wk"].to_list() + [0.0] * 999)[starters[p] - 1] for p in POS}
    d = d.with_columns((pl.col("wk") - pl.col("position").replace_strict(repl, return_dtype=pl.Float64)).alias("por"))
    parts = []
    for p in POS:
        s = d.filter(pl.col("position") == p).sort("wk", descending=True)
        parts.append(s.with_columns(pl.int_range(1, pl.len() + 1).alias("pos_rank"),
                                    pl.Series("tier", _tiers(s["wk"], *TIER_SHAPE[p]))))
    b = pl.concat(parts).sort("por", descending=True).with_columns(pl.int_range(1, pl.len() + 1).alias("rk"))
    b = b.join(ros.role_flags(season, thru), on="player_id", how="left").with_columns(
        pl.when(pl.col("position") == "DEF").then(pl.col("team").replace_strict(TEAMS, default=pl.col("player_display_name")))
          .otherwise(pl.col("player_display_name")).alias("player_display_name"))
    r1 = lambda c: pl.col(c).round(1)
    df = b.select(
        pl.col("rk").alias("Overall Rk"), pl.col("position").alias("Pos"), pl.col("pos_rank").alias("Pos Rk"),
        pl.col("tier").alias("Pos Tier"), pl.col("player_display_name").alias("Player"), pl.col("team").alias("Team"),
        pl.when(pl.col("home")).then(pl.col("opp")).otherwise(pl.concat_str([pl.lit("@"), pl.col("opp")])).alias("Opp"),
        pl.concat_str([pl.col("gameday").str.slice(5), pl.lit(" "), pl.col("gametime")]).alias("Kickoff"),
        r1("wk").alias("Wk Proj"),
        pl.when(pl.col("status") == "season").then(pl.lit("Out for season"))
          .when(pl.col("news").fill_null("") != "").then(pl.col("news").str.split(",").list.first())
          .when(pl.col("status") == "practice").then(pl.lit("Practice squad"))
          .when(pl.col("status") == "no_games").then(pl.lit("No games yet"))
          .when(pl.col("status") == "out_long").then(pl.lit("IR / PUP / suspended"))
          .when(pl.col("status") == "out_week").then(pl.lit("Out"))
          .otherwise(pl.lit("")).alias("Status"),
        (pl.col("play") * 100).round(0).cast(pl.Int32).alias("Play Chance"),
        r1("base").alias("Base PPG"), pl.col("match_adj").round(2).alias("Matchup Adj"),
        pl.when(pl.col("position") == "DEF").then(pl.col("opp_imp")).otherwise(pl.col("imp")).round(1).alias("Team Implied Pts"),
        pl.col("vegas_adj").round(2).alias("Vegas Adj"),
        pl.col("absence_adj").round(2).alias("Teammate Out Adj"),
        pl.col("role_flag").fill_null("").alias("Role"),
        r1(f"ppg_{lg}_sofar").alias("PPG So Far"), r1("l3").alias("Last 3 PPG"), pl.col("n").alias("Games Played"),
        r1("por").alias("Pts Over Repl"), pl.col("news").fill_null("").alias("Injury News"),
        pl.when(pl.col("rookie")).then(pl.lit("Rookie")).otherwise(pl.lit("")).alias("Note"),
        pl.col("player_id").alias("nflverse ID"))
    # keep the sheets readable: players who could plausibly start
    df = df.filter((pl.col("Wk Proj") >= 1.0) | (pl.col("Pos Rk") <= 40))
    wb = Workbook()

    def sheet(ws, frame, freeze):
        ws.append(frame.columns)
        for c in ws[1]:
            c.font, c.fill = Font(bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864")
        for row in frame.iter_rows():
            ws.append(list(row))
        ws.freeze_panes = freeze
        for c in ws.columns:
            ws.column_dimensions[c[0].column_letter].width = max(9, min(26, max(len(str(x.value or "")) for x in c[:60]) + 2))

    ws = wb.active
    ws.title = "Overall"
    sheet(ws, df, "F2")
    for p in POS:
        sub = df.filter(pl.col("Pos") == p).sort("Pos Rk").drop("Pos")
        sheet(wb.create_sheet(p), sub.select(["Pos Rk"] + [c for c in sub.columns if c != "Pos Rk"]), "D2")
    fx = df.filter(pl.col("Pos").is_in(L["flex_eligible"])).sort("Wk Proj", descending=True).with_columns(
        pl.int_range(1, pl.len() + 1).alias("Flex Rk"))
    sheet(wb.create_sheet("FLEX"), fx.select(["Flex Rk"] + [c for c in fx.columns if c not in ("Flex Rk", "Overall Rk", "Pos Tier")]), "E2")
    ws = wb.create_sheet("Notes")
    for line in [
        f"{'THE BURBS' if lg == 'burbs' else 'CBS'}: week {week} rankings, {season}, using games through week {thru}.",
        f"On bye: {', '.join(byes) if byes else 'none'}.",
        "Wk Proj = Base PPG (rest-of-season talent/role, schedule removed) + Matchup Adj + Vegas Adj, times Play Chance.",
        "Matchup: this opponent's points allowed to the position this season vs what those players usually score.",
        "Vegas: team implied points this week vs its usual this season. Weights fitted on 2021-2025 weeks 5-17:",
        "  matchup x1.0 QB, x0.45 RB, x0.35 WR, x0.55 TE, x0.5 DEF; Vegas +0.075/pt RB and TE, +0.13/pt K, -0.25/pt of opponent's total for DEF.",
        "  Honest caveat: matchups and lines explain only a small part of one week's score (week-to-week noise is large);",
        "  they mattered most for defenses. Use them as tiebreakers between similar players, not to bench a stud.",
        "Teammate out: if a team's top WR may sit, other WRs get up to +2.0 and the top TE +0.9 (history 2016-2025), scaled by the chance he sits.",
        "Injuries: from the latest injury report you pasted. Out / Doubtful / IR = 0; Questionable = 85% chance to play (65% if he did not practice Wednesday); concussion protocol = 55%.",
        "  Paste Friday's or Saturday's report and these refresh with final designations.",
        "Overall tab is ranked by points over the last starter at the position in a 12-team league (for trades of value across positions,",
        "  use the ROS file instead). FLEX tab ranks RB/WR/TE together by Wk Proj.",
        f"Rerun: PYTHONPATH=src python3 scripts/build_weekly.py {season} {week}",
    ]:
        ws.append([line])
    ws.column_dimensions["A"].width = 120
    path = out_dir / f"weekly_{season}_wk{week}_{lg}.xlsx"
    wb.save(path)
    add_guide(path)
    print(path, df.height, "byes:", byes)
