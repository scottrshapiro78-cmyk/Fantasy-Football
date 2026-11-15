"""Compare a pasted expert rest-of-season list (Footballguys) with the model's ROS file.
    PYTHONPATH=src python3 scripts/compare_expert.py data/input/expert/fbg_ros_2026wk10_burbs.csv burbs 10
"""
import re, sys
import numpy as np, pandas as pd
from ffmodel.config import DATA
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200); pd.set_option('display.max_colwidth', 45)
path, lg, week = sys.argv[1], sys.argv[2], int(sys.argv[3])
def norm(s):
    s = re.sub(r"[.'’]", "", s.lower()); s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = s.replace("ken walker", "kenneth walker").replace("kc concepcion", "kevin concepcion")
    return re.sub(r"\s+", " ", s).strip()
e = pd.read_csv(path); e["pos"] = e.pos.replace({"TD": "DEF", "PK": "K"}); e["key"] = e.player.map(norm)
m = pd.read_excel(DATA / "output" / f"ros_2026_wk{week}_{lg}.xlsx", sheet_name="Overall"); m["key"] = m.Player.map(norm)
d = e.merge(m[["key", "Pos", "Player", "Team", "Status", "ROS PPG", "PPG So Far", "Preseason PPG", "Games Played", "Injury News", "Pos Rk"]],
            left_on=["key", "pos"], right_on=["key", "Pos"], how="left")
print("unmatched:", d[d.Player.isna()].player.tolist())
d = d[d.Player.notna()].copy()
# their horizon: weeks 10-18 (9 games) fits ROS / This Week for healthy players better than 8
bye_left = (d.bye.astype(int) >= week).astype(int)
out_now = (d.this_week == 0) & (d.bye.astype(int) != week)
d["fbg_games"] = 9 - bye_left
d["fbg_ppg"] = d.ros / d.fbg_games
healthy = d[~out_now & d.Status.isna()].copy()
print("ratio ROS/this week, healthy:", (healthy.ros / healthy.this_week.replace(0, np.nan)).median().round(2))
healthy["diff"] = healthy["ROS PPG"] - healthy.fbg_ppg
healthy["fbg_pos_rk"] = healthy.groupby("pos").fbg_ppg.rank(ascending=False).astype(int)
healthy["my_pos_rk"] = healthy.groupby("pos")["ROS PPG"].rank(ascending=False).astype(int)
c = ["pos", "Player", "Team", "fbg_ppg", "ROS PPG", "diff", "PPG So Far", "Preseason PPG", "Games Played", "fbg_pos_rk", "my_pos_rk"]
for p in ["QB", "RB", "WR", "TE", "K", "DEF"]:
    h = healthy[healthy.pos == p]
    print(f"\n{p}: n={len(h)} mean diff (me - them) {h['diff'].mean():.2f}, corr {h['ROS PPG'].corr(h.fbg_ppg):.2f}, rank corr {h['ROS PPG'].corr(h.fbg_ppg, method='spearman'):.2f}")
sk = healthy[healthy.pos.isin(["QB", "RB", "WR", "TE"])]
print("\nMODEL LOWER than FBG"); print(sk.sort_values("diff").head(22)[c].round(1).to_string(index=False))
print("\nMODEL HIGHER than FBG"); print(sk.sort_values("diff", ascending=False).head(15)[c].round(1).to_string(index=False))
inj = d[out_now | d.Status.notna()]
print("\nINJURED / OUT THIS WEEK (their ROS total vs my status)")
print(inj[["pos", "Player", "this_week", "ros", "ROS PPG", "Status", "Injury News"]].round(1).to_string(index=False))
healthy.to_csv(DATA / "output" / f"expert_compare_wk{week}_{lg}.csv", index=False)
