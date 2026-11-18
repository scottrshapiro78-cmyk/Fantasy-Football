import polars as pl, numpy as np
X=pl.read_parquet('scratch/X.parquet')
K=4
def rate(df,p,c):
    mu=df.filter(pl.col('season')<2026)[c].mean() if c in df.columns else None
    return ((pl.col(p+c).fill_null(0)+K*mu)/(pl.col(p+'n')+K))
cols=['kb','range_dr','rz_stall','pat','fga','rz_dr','drives']
X=X.with_columns(*[rate(X,'o_',c).alias('or_'+c) for c in cols],*[rate(X,'d_',c).alias('dr_'+c) for c in cols])
X=X.filter(pl.col('week')>=4)
sets={'M0 team K pts to date':['or_kb'],
 'M1 + offense opportunity':['or_kb','or_range_dr','or_rz_stall','or_pat'],
 'M2 + opp defense allowed':['or_kb','or_range_dr','or_rz_stall','or_pat','dr_kb','dr_range_dr','dr_rz_stall','dr_pat'],
 'M2b defense K-pts only':['or_kb','or_range_dr','or_rz_stall','or_pat','dr_kb'],
 'M3 M1 + venue (indoor/cold/home)':['or_kb','or_range_dr','or_rz_stall','or_pat','indoor','cold','home'],
 'M4 M2 + venue':['or_kb','or_range_dr','or_rz_stall','or_pat','dr_kb','dr_range_dr','dr_rz_stall','dr_pat','indoor','cold','home']}
def fit(tr,f,y):
    A=np.c_[np.ones(len(tr)),tr.select(f).to_numpy()]; return np.linalg.lstsq(A,tr[y].to_numpy(),rcond=None)[0]
def pred(te,f,b): return np.c_[np.ones(len(te)),te.select(f).to_numpy()]@b
for y in ['kb','kc']:
  print('target',y)
  for name,f in sets.items():
    errs=[];per={}
    for s in range(2021,2027):
        tr=X.filter(pl.col('season')!=s,pl.col('season')<2026) if s<2026 else X.filter(pl.col('season')<2026)
        te=X.filter(pl.col('season')==s); b=fit(tr,f,y)
        e=np.abs(te[y].to_numpy()-pred(te,f,b)); per[s]=e.mean()
        if s<2026: errs.append(e)
    print(f"  {name:34s} LOSO MAE 21-25 {np.concatenate(errs).mean():.4f}  2026 {per[2026]:.4f}  by yr "+' '.join(f'{per[s]:.3f}' for s in range(2021,2026)))
b=fit(X.filter(pl.col('season')<2026),sets['M4 M2 + venue'],'kb'); print('M4 coefs',dict(zip(['c']+sets['M4 M2 + venue'],np.round(b,3))))
b=fit(X.filter(pl.col('season')<2026),sets['M3 M1 + venue (indoor/cold/home)'],'kb'); print('M3 coefs',dict(zip(['c']+sets['M3 M1 + venue (indoor/cold/home)'],np.round(b,3))))
# simple descriptive: kicker pts by opponent defense tercile (to-date allowed), all 2021-25
d=X.filter(pl.col('season')<2026,pl.col('week')>=6).with_columns(pl.col('dr_kb').qcut(3,labels=['tough','mid','soft']).alias('dt'))
print(d.group_by('dt').agg(pl.len(),pl.col('kb').mean(),pl.col('or_kb').mean()).sort('dt'))
print(X.filter(pl.col('season')<2026).group_by('indoor','cold').agg(pl.len(),pl.col('kb').mean(),(pl.col('fga')).mean()).sort('indoor','cold'))
X.write_parquet('scratch/Xr.parquet')
