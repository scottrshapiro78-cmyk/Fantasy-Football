import json, numpy as np, polars as pl
pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(220)
f=json.load(open('data/processed/ros_fit.json'))
h=pl.concat([pl.read_parquet(f'data/processed/ros_history_wk{t}.parquet').with_columns(pl.lit(t).alias('thru')) for t in (8,9,10)],how='diagonal_relaxed')
def run(lg, ratio=1.5, min_n=6):
    out=[]
    for p in ('RB','WR','TE'):
        d=h.filter(pl.col('position')==p, pl.col('n_after')>=3, pl.col('n')>=min_n, pl.col(f'pred_ppg_{lg}').is_not_null(), pl.col('xfp_pg').is_not_null())
        k,u,r,s=f['k'][lg][p],f['u'][lg][p],f['r'][lg][p],f['s'][lg][p]
        pri=d[f'pred_ppg_{lg}'].to_numpy(); n=d['n'].to_numpy(); so=d[f'ppg_{lg}_sofar'].to_numpy()
        luck=d['luck_pg'].fill_null(0).to_numpy(); rt=d['role_trend'].fill_null(0).to_numpy(); xf=d['xfp_pg'].to_numpy()
        sf=d[f'sos_faced_{lg}'].fill_null(0).to_numpy(); sl=d[f'sos_left_{lg}'].fill_null(0).to_numpy(); y=d[f'ppg_{lg}_after'].to_numpy()
        obs=so+u*luck+r*rt-s*sf
        def pred(kk, prior=pri): return (kk*prior+n*obs)/(kk+n)+s*sl
        import os; brk=(xf>=ratio*pri)&(xf>=float(os.environ.get("MINX","6")))
        V={'current':pred(k),'k x0.5':pred(k*.5),'k x0.25':pred(k*.25),'k=0 (season only)':pred(0*pri),
           'prior=avg(prior,xfp)':pred(k,(pri+xf)/2),'prior=xfp':pred(k,xf),'season avg':so,'xfp only':xf}
        for name,v in V.items():
            out.append((p,name,int(brk.sum()),float(np.mean(np.abs(v-y)[brk])),float(np.mean((v-y)[brk])),float(np.mean(np.abs(V['current']-y)[~brk]))))
        out.append((p,'_seasons',0,0,0,0))
        for se in sorted(set(d['season'].to_list())):
            m=brk&(d['season'].to_numpy()==se)
            out.append((p,f'  {se}: current vs prior=xfp vs kx0.25',int(m.sum()),float(np.mean(np.abs(V['current']-y)[m])),float(np.mean(np.abs(V['prior=xfp']-y)[m])),float(np.mean(np.abs(V['k x0.25']-y)[m]))))
    return pl.DataFrame(out,schema=['pos','variant','n','mae','bias','mae_rest_current'],orient='row')
import sys
print(run(sys.argv[1], float(sys.argv[2])).with_columns(pl.col(pl.Float64).round(2)))
