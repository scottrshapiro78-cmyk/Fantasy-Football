import polars as pl, numpy as np, nflreadpy as nfl, re
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
X=pl.read_parquet('data/processed/kicker_Xr.parquet'); tg=pl.read_parquet('data/processed/kicker_tg.parquet')
f=['or_kb','or_range_dr','or_rz_stall','or_pat','indoor','cold','home']
tr=X.filter(pl.col('season')<2026)
B={}
for y in ['kb','kc']:
    A=np.c_[np.ones(len(tr)),tr.select(f).to_numpy()]; B[y]=np.linalg.lstsq(A,tr[y].to_numpy(),rcond=None)[0]
K=4; t26=tg.filter(pl.col('season')==2026)
mu={c:tg.filter(pl.col('season')<2026)[c].mean() for c in ['kb','range_dr','rz_stall','pat']}
team=t26.group_by('posteam').agg(pl.len().alias('g'),*[pl.col(c).sum().alias('s_'+c) for c in ['kb','kc','range_dr','rz_dr','rz_stall','fga','pat']])
team=team.with_columns(*[((pl.col('s_'+c)+K*mu[c])/(pl.col('g')+K)).alias('or_'+c) for c in mu])
s26=nfl.load_schedules([2026]).filter(pl.col('game_type')=='REG')
hr=s26.filter(pl.col('roof').is_not_null()).group_by('home_team').agg(pl.col('roof').mode().first().alias('hroof'))
sch=s26.filter(pl.col('week')>=11,pl.col('week')<=17).join(hr,on='home_team',how='left').with_columns(pl.col('roof').fill_null(pl.col('hroof')))
print(sch['roof'].value_counts())
g=pl.concat([sch.select('week','gameday','roof',pl.col('home_team').alias('posteam'),pl.lit(1).alias('home')),
             sch.select('week','gameday','roof',pl.col('away_team').alias('posteam'),pl.lit(0).alias('home'))]).with_columns(
   pl.col('roof').is_in(['dome','closed']).cast(pl.Int8).alias('indoor'),
   (pl.col('roof').is_in(['outdoors','open'])&pl.col('gameday').str.slice(5,2).is_in(['12','01'])).cast(pl.Int8).alias('cold'))
g=g.join(team,on='posteam')
for y in ['kb','kc']:
    g=g.with_columns(pl.Series('p_'+y,np.c_[np.ones(len(g)),g.select(f).to_numpy()]@B[y]))
r=g.group_by('posteam').agg(pl.len().alias('gl'),pl.col('p_kb').sum().alias('ros_burbs'),pl.col('p_kc').sum().alias('ros_cbs'),
     pl.col('p_kb').filter(pl.col('week')>=15).sum().alias('wk15_17_burbs'),pl.col('indoor').sum().alias('indoor_g'),
     pl.col('week').filter(pl.col('week')>=15).len().alias('g15'))
r=r.join(team.select('posteam','g',(pl.col('s_kb')/pl.col('g')).alias('ppg_burbs'),(pl.col('s_range_dr')/pl.col('g')).alias('range_dr_pg'),
     (pl.col('s_rz_stall')/pl.col('s_rz_dr')).alias('rz_stall_pct'),(pl.col('s_fga')/pl.col('g')).alias('fga_pg'),(pl.col('s_pat')/pl.col('g')).alias('pat_pg')),on='posteam')
# current kicker = kicker in team's most recent game
last=t26.sort('week').group_by('posteam').agg(pl.col('kid').last(),pl.col('kname').last(),pl.col('week').last().alias('last_wk'),
      pl.col('kid').unique().len().alias('n_k'))
rw=nfl.load_rosters_weekly([2026]); mw=rw['week'].max()
rk=rw.filter(pl.col('week')==mw,pl.col('position')=='K').select(pl.col('gsis_id').alias('kid'),'full_name','team','status')
inj=nfl.load_injuries([2026]); iw=inj['week'].max()
ik=inj.filter(pl.col('week')==iw,pl.col('position')=='K').select(pl.col('gsis_id').alias('kid'),'report_status')
ps=nfl.load_player_stats([2026]).filter(pl.col('position')=='K').group_by('player_id').agg(pl.len().alias('k_games26'),pl.col('team').last().alias('stat_team')).rename({'player_id':'kid'})
r=r.join(last,on='posteam').join(rk,on='kid',how='left').join(ik,on='kid',how='left').join(ps,on='kid',how='left')
def norm(s): return re.sub(r'[^a-z]','',s.lower().replace(' jr','').replace(' iii',''))
def held(path):
    d=pl.read_csv(path); return {norm(p):t for p,t,s in zip(d['Player'],d['Team'],d['Slot']) if s=='K' or True}
bu=held('/mnt/project-files/ff-model/burbs_rosters_2026wk11.csv'); cb=held('/mnt/project-files/ff-model/cbs_rosters_2026wk11.csv')
r=r.with_columns(pl.col('full_name').fill_null(pl.col('kname')).alias('kicker')).with_columns(
   pl.col('kicker').map_elements(lambda n: bu.get(norm(n),'FA'),return_dtype=pl.Utf8).alias('burbs'),
   pl.col('kicker').map_elements(lambda n: cb.get(norm(n),'FA'),return_dtype=pl.Utf8).alias('cbs'))
r=r.sort('ros_burbs',descending=True).with_row_index('rk',1)
print('roster week',mw,'injury week',iw)
print(r.select('rk','kicker','posteam','team','status','report_status','k_games26','last_wk','n_k','gl','g15','indoor_g',
  pl.col('ros_burbs').round(1),pl.col('ros_cbs').round(1),pl.col('wk15_17_burbs').round(1),pl.col('ppg_burbs').round(1),
  pl.col('range_dr_pg').round(2),pl.col('rz_stall_pct').round(2),pl.col('fga_pg').round(2),pl.col('pat_pg').round(2),'burbs','cbs'))
r.write_csv('data/processed/kicker_ros_k.csv')

# ---- verified current kicker per team
rwk=rw.filter(pl.col('position')=='K')
lastk=rwk.sort('week').group_by('gsis_id').agg(pl.col('team').last().alias('rteam'),pl.col('status').last().alias('rstatus'),pl.col('week').last().alias('rweek'),pl.col('full_name').last())
fin=r.select('posteam','kid','kname','ros_burbs','ros_cbs','wk15_17_burbs','ppg_burbs','range_dr_pg','rz_stall_pct','fga_pg','pat_pg','gl','k_games26').join(lastk.rename({'gsis_id':'kid'}),on='kid',how='left')
notes={'MIA':'Gonzalez listed CUT; Carlson on practice squad. MIA kicker unsettled','LA':'Mevis now on LA active roster; McManus not on Week 11 list. Check before adding',
 'NO':'Smyth is a practice-squad elevation','ATL':'Folk cut; Moody kicked Week 10','BAL':'Loop on IR; Patterson kicked Week 10'}
fin=fin.with_columns(pl.col('posteam').replace_strict(notes,default='').alias('note'),pl.col('full_name').fill_null(pl.col('kname')).alias('kicker')).with_columns(
   pl.col('kicker').map_elements(lambda n: bu.get(norm(n),'FA'),return_dtype=pl.Utf8).alias('burbs'),
   pl.col('kicker').map_elements(lambda n: cb.get(norm(n),'FA'),return_dtype=pl.Utf8).alias('cbs'))
fin=fin.sort('ros_burbs',descending=True).with_row_index('rank',1)
out=fin.select('rank','kicker',pl.col('posteam').alias('team'),pl.col('gl').alias('games_left'),pl.col('ros_burbs').round(1),pl.col('ros_cbs').round(1),
  pl.col('wk15_17_burbs').round(1),pl.col('ppg_burbs').round(1).alias('ppg_2026'),pl.col('range_dr_pg').round(2).alias('drives_to_opp30_pg'),
  pl.col('rz_stall_pct').round(2).alias('rz_fg_rate'),pl.col('fga_pg').round(2),pl.col('pat_pg').round(2),'k_games26',pl.col('rstatus').alias('nfl_status'),'rweek','burbs','cbs','note')
print(out.drop('drives_to_opp30_pg','rz_fg_rate','fga_pg','pat_pg'))
import os; os.makedirs('/mnt/project-files/ff-model/kickers',exist_ok=True)
out.write_csv('/mnt/project-files/ff-model/kickers/kicker_ros_2026_wk11.csv')
