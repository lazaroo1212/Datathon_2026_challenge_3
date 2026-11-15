"""Uso: python3 train_v1.py exclude|all salida.json ruta/hourly.csv
Entrena tablas de perfil + ridge por bucket de horizonte para w, load y pv12. exclude=True deja fuera dias 18-24 (eval honesto)."""
import numpy as np, pandas as pd, json, sys
PVS=12.0; G=0.27
def build(exclude, out, hourly):
    df=pd.read_csv(hourly,parse_dates=["Fecha"]).set_index("Fecha")
    full=pd.date_range(df.index[0],df.index[-1],freq="1h"); df=df.reindex(full)
    load=df.load_kw.values.copy(); pv=df.pv_kw.values*PVS
    w=(df.co2_kg_per_kwh+G*df.precio_eur_per_kwh).values
    idx=df.index; mon=idx.month.values-1; hr=idx.hour.values; dt=(idx.dayofweek.values>=5).astype(int)
    inwin=(idx.day.values>=18)&(idx.day.values<=24)
    use=(~inwin) if exclude else np.ones(len(idx),bool)
    def prof(x):
        P=np.full((12,24,2),np.nan)
        for m in range(12):
            for h in range(24):
                for d in range(2):
                    s=(mon==m)&(hr==h)&(dt==d)&use&~np.isnan(x)
                    if s.sum()>=3: P[m,h,d]=x[s].mean()
        # fallback: month,hour mean ; then hour mean
        for m in range(12):
            for h in range(24):
                for d in range(2):
                    if np.isnan(P[m,h,d]):
                        s=(mon==m)&(hr==h)&use&~np.isnan(x)
                        if s.sum()>=3: P[m,h,d]=x[s].mean()
                        else:
                            s=(hr==h)&use&~np.isnan(x); P[m,h,d]=x[s].mean()
        return P
    PL,PP,PW=prof(load),prof(pv),prof(w)
    pl=PL[mon,hr,dt]; pp=PP[mon,hr,dt]; pw=PW[mon,hr,dt]
    n=len(idx)
    def ok(*a): return not any(np.isnan(v) for v in a)
    rows={"load":[], "pv":[], "w":[]}
    for t in range(170,n-30):
        if not use[t]: continue
        # ratios (info up to t-1 for load/pv; up to t+2 for w)
        rL=np.clip(load[t-1]/pl[t-1],0.4,1.8) if ok(load[t-1]) and pl[t-1]>1 else 1.0
        rP=np.clip(pv[t-1]/pp[t-1],0,2.5) if ok(pv[t-1]) and pp[t-1]>20 else 1.0
        kw=w[t:t+3]; 
        rW=(np.mean(kw)/np.mean(pw[t:t+3])) if ok(*kw) else None
        for k in range(0,24):
            T=t+k
            if not use[T]: continue
            if ok(load[T]) and ok(load[T-24]): rows["load"].append((k,load[T],1,pl[T],pl[T]*rL,load[T-24]))
            if ok(pv[T]) and ok(pv[T-24]) and pp[T]>1: rows["pv"].append((k,pv[T],1,pp[T],pp[T]*rP,pv[T-24]))
        if rW is not None:
            for h in range(3,27):
                T=t+h
                if use[T] and ok(w[T],w[T-24],w[t+2]): rows["w"].append((h,w[T],1,pw[T],pw[T]*rW,w[T-24],w[t+2]))
    coef={}
    bk={"load":[(0,2),(3,5),(6,11),(12,23)],"pv":[(0,2),(3,5),(6,11),(12,23)],"w":[(3,6),(7,12),(13,20),(21,26)]}
    for key in rows:
        A=np.array(rows[key]); coef[key]=[]
        for lo,hi in bk[key]:
            s=(A[:,0]>=lo)&(A[:,0]<=hi); X=A[s,2:]; y=A[s,1]
            b=np.linalg.solve(X.T@X+1e-3*np.eye(X.shape[1]),X.T@y)
            coef[key].append(dict(lo=lo,hi=hi,b=b.tolist()))
            print(key,(lo,hi),len(y),np.round(b,3).tolist())
    params=dict(PL=PL.tolist(),PP=PP.tolist(),PW=PW.tolist(),coef=coef,G=G,pv_scale=PVS,
                gamma_default=0.27,horizon=24,ext_known=3)
    json.dump(params,open(out,"w"))
if __name__=="__main__":
    build(sys.argv[1]=="exclude", sys.argv[2], sys.argv[3])
