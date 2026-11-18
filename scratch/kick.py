import nflreadpy as nfl, polars as pl, numpy as np
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(250)
SEAS=list(range(2021,2027))
pbp=nfl.load_pbp(SEAS).filter(pl.col('season_type')=='REG', pl.col('posteam').is_not_null())
sch=nfl.load_schedules(SEAS).filter(pl.col('game_type')=='REG')
fg=pl.col('field_goal_attempt')==1; xp=pl.col('extra_point_attempt')==1
made=pl.col('field_goal_result')=='made'; d=pl.col('kick_distance')
fgpts=pl.when(fg&made).then(pl.when(d>=50).then(5).when(d>=40).then(4).otherwise(3)).otherwise(0)
xpg=pl.col('extra_point_result')=='good'
pbp=pbp.with_columns(
  (fgpts+pl.when(xp&xpg).then(1).when(xp).then(-1).otherwise(0)).alias('kb'),
  (fgpts+pl.when(xp&xpg).then(1).otherwise(0)).alias('kc'))
# drive-level
dr=(pbp.filter(pl.col('fixed_drive').is_not_null()).group_by('game_id','posteam','fixed_drive')
    .agg(pl.col('yardline_100').min().alias('best'),pl.col('fixed_drive_result').first().alias('res')))
dr=dr.group_by('game_id','posteam').agg(pl.len().alias('drives'),(pl.col('best')<=30).sum().alias('range_dr'),
   (pl.col('best')<=20).sum().alias('rz_dr'),
   ((pl.col('best')<=20)&pl.col('res').is_in(['Field goal','Missed field goal'])).sum().alias('rz_stall'))
tg=(pbp.group_by('game_id','season','week','posteam','defteam')
    .agg(pl.col('kb').sum(),pl.col('kc').sum(),fg.sum().alias('fga'),(fg&made).sum().alias('fgm'),xp.sum().alias('pat'),
         pl.col('kicker_player_id').filter(fg|xp).drop_nulls().mode().first().alias('kid'),pl.col('kicker_player_name').filter(fg|xp).drop_nulls().mode().first().alias('kname'))
    .join(dr,on=['game_id','posteam']))
# venue
ven=sch.select('game_id','roof','temp','wind','home_team','gameday')
tg=tg.join(ven,on='game_id',how='left').with_columns(
   pl.col('roof').is_in(['dome','closed']).cast(pl.Int8).alias('indoor'),
   (pl.col('roof').is_in(['outdoors','open'])&pl.col('gameday').str.slice(5,2).is_in(['12','01'])).cast(pl.Int8).alias('cold'),
   (pl.col('posteam')==pl.col('home_team')).cast(pl.Int8).alias('home'))
tg.write_parquet('scratch/tg.parquet')
# to-date (prior weeks, same season) rates for offense and defense
def todate(df,key,cols,pref):
    df=df.sort('season',key,'week')
    out=df.select('season','week',key,*[ (pl.col(c).cum_sum().shift(1).over('season',key)).alias(pref+c) for c in cols],
                  pl.int_range(pl.len()).over('season',key).alias(pref+'n'))
    return out
cols=['kb','kc','range_dr','rz_dr','rz_stall','fga','pat','drives']
off=todate(tg,'posteam',cols,'o_')
de=todate(tg,'defteam',cols,'d_')
X=tg.join(off,on=['season','week','posteam']).join(de,on=['season','week','defteam'])
X.write_parquet('scratch/X.parquet')
print(tg.group_by('season').agg(pl.len(),pl.col('kb').mean(),pl.col('range_dr').mean(),pl.col('indoor').mean()).sort('season'))
