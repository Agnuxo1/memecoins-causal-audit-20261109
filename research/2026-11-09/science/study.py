"""Frozen causal daily comparison. Francisco Angulo de Lafuente, 2026.

No network, fitted artefact, test outcomes, GPU or row ranking. Read only supplied
training targets. Each held-out day is treated as a completely unlabelled batch.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from legacy_features import creator_rep, wallet_rep, HolographicReservoir

LEVELS=np.array([3.,4.,5.,7.,10.,15.,20.])
SEEDS=(0,1,2)
GRID=np.round(np.arange(0.,2.51,.1),2)
WIN=3600
EMBARGO=3600
LGB=dict(n_estimators=400,learning_rate=.04,num_leaves=31,min_child_samples=60,
         subsample=.85,subsample_freq=1,colsample_bytree=.7,reg_lambda=5.,
         verbosity=-1,n_jobs=4,deterministic=True,force_col_wise=True)
SKIP={'mint','mt','max_x','final_x','day','created_ts','creator'}
REP_PREFIX='r_'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def atomic_npz(path,**arrays):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def pnl(tp,mx,fx):
    tp=np.asarray(tp,float); mx=np.asarray(mx,float); fx=np.asarray(fx,float)
    if not np.isfinite(tp).all() or not ((tp==0)|((tp>=3)&(tp<=100))).all():
        raise ValueError('Illegal prediction')
    return np.where(tp==0,0.,np.where(mx>=tp,tp-1.3,np.maximum(fx-1.3,-1.)))

class CausalPanel:
    def __init__(self,panel,edges,T):
        self.a=panel.reset_index(drop=True).copy()
        if self.a.mint.duplicated().any() or not self.a.mt.is_monotonic_increasing:
            raise ValueError('Panel must have unique mints and stable chronological order')
        if T.shape!=(len(self.a),20,7) or not np.isfinite(T).all(): raise ValueError('Bad sequence alignment/values')
        self.T=np.asarray(T,np.float32)
        self.base=[c for c in self.a if c not in SKIP and not c.startswith(REP_PREFIX)
                   and self.a[c].dtype.kind in 'fiub']
        self.cre=self.a.creator.fillna('?').astype('category').cat.codes.to_numpy()
        self.by=[None]*len(self.a); self.nw=0
        if edges is not None and len(edges):
            # Factorise identities once, not once per daily view; IDs never enter X.
            wid=pd.factorize(edges.wallet,sort=False)[0].astype(np.int32)
            mi=pd.Index(self.a.mint).get_indexer(edges.mint)
            sol=edges.sol.to_numpy(dtype=float)
            keep=mi>=0; mi=mi[keep]; wid=wid[keep]; sol=sol[keep]
            order=np.argsort(mi,kind='stable');mi=mi[order];wid=wid[order];sol=sol[order]
            if len(mi):
                ids,starts=np.unique(mi,return_index=True); ends=np.r_[starts[1:],len(mi)]
                for i,b,e in zip(ids,starts,ends):
                    if len(np.unique(wid[b:e]))!=e-b: raise ValueError('Duplicate mint/wallet edge')
                    self.by[int(i)]=(wid[b:e],sol[b:e])
                self.nw=int(wid.max())+1
        self.rep=self.reputation()
        self.train=self.a.drop(columns=[c for c in self.a if c.startswith(REP_PREFIX)])
        self.train=pd.concat([self.train,self.rep],axis=1)
        self.fe=self.base+list(self.rep.columns)
        self.day_cache={}
        # Fixed random recurrent states: no readout fitting, labels or global scaling.
        self.state=np.zeros((len(self.a),96),dtype=np.float64)
        for s in SEEDS:
            self.state+=HolographicReservoir(size=48,input_dim=5,seed=s).states(self.T[:,:,[0,1,2,3,6]])[:,:96]/len(SEEDS)
        if not np.isfinite(self.state).all(): raise ValueError('Nonfinite reservoir state')

    def reputation(self,cutoff=None):
        """Same statistics as original functions, with pre-indexed wallet edges."""
        a=self.a; mts=a.mt.to_numpy(); mx=a.max_x.to_numpy()
        valid=np.isfinite(mx)
        if cutoff is not None: valid &= mts<=cutoff
        nc=int(self.cre.max())+1
        n=np.zeros(nc); hits=np.zeros((nc,3))
        seen=np.zeros(self.nw,np.int32); whits=np.zeros(self.nw,np.int32)
        out=np.full((len(a),11),np.nan); j=0
        for i in range(len(a)):
            while j<len(a) and mts[j]<=mts[i]-WIN:
                if valid[j]:
                    c=self.cre[j];n[c]+=1;hits[c]+=(mx[j]>=np.array([3.,5.,10.]))
                    w=self.by[j]
                    if w is not None:
                        seen[w[0]]+=1
                        if mx[j]>=5: whits[w[0]]+=1
                j+=1
            c=self.cre[i];k=n[c];out[i,0]=k
            if k>0: out[i,1:4]=hits[c]/k
            w=self.by[i]
            if w is None: continue
            wid,sol=w;s=seen[wid];kn=s>=3;out[i,10]=len(wid)
            if not kn.any(): out[i,4:10]=0.;continue
            rate=whits[wid][kn]/s[kn];elite=(s>=5)&(whits[wid]>=1)
            out[i,4:10]=[kn.sum(),rate.max(),rate.mean(),np.sort(rate)[-3:].mean(),
                         elite.sum(),sol[elite].sum()/max(sol.sum(),1e-9)]
        cols=['r_creator_n','r_creator_p3','r_creator_p5','r_creator_p10','r_w_known',
              'r_w_max','r_w_mean','r_w_top3','r_w_elite_n','r_w_elite_share','r_w_nwal']
        return pd.DataFrame(out,columns=cols,index=a.index)

    def day_view(self,day):
        if day not in self.day_cache:
            mask=(self.a.day==day).to_numpy()
            if not mask.any(): raise ValueError('Day absent')
            cutoff=float(self.a.loc[mask,'mt'].min())-WIN-EMBARGO
            rep=self.reputation(cutoff)
            df=self.a.loc[mask].drop(columns=[c for c in self.a if c.startswith(REP_PREFIX)]).copy()
            for c in rep: df[c]=rep.loc[mask,c].to_numpy()
            self.day_cache[day]=df
        return self.day_cache[day].copy()

    def matrix(self,df,arm):
        X=df[self.fe].to_numpy(dtype=float,copy=True)
        X[~np.isfinite(X)]=np.nan
        if arm=='reservoir': X=np.concatenate([X,self.state[df.index.to_numpy()]],axis=1)
        elif arm!='control': raise ValueError('Unknown arm')
        return X

def predict_day(data,day,arm):
    val=data.day_view(day)
    cutoff=float(val.mt.min())-WIN-EMBARGO
    tr=data.train[data.train.max_x.notna() & (data.train.mt<=cutoff)].copy()
    days=sorted(tr.day.unique())
    if len(days)<12 or len(tr)<3000: raise ValueError('Insufficient training history')
    caldays=days[-5:]
    cal=pd.concat([data.day_view(d) for d in caldays]).loc[lambda x:x.index.isin(tr.index)]
    fit=tr[(tr.day<caldays[0]) & (tr.mt<=float(cal.mt.min())-WIN-EMBARGO)]
    if len(fit)<2000 or len(cal)<200: raise ValueError('Insufficient disjoint calibration')
    if float(fit.mt.max())>float(cal.mt.min())-WIN-EMBARGO: raise RuntimeError('Fit/calibration overlap')
    if float(tr.mt.max())>float(val.mt.min())-WIN-EMBARGO: raise RuntimeError('Unresolved labels')
    Xf=data.matrix(fit,arm); Xc=data.matrix(cal,arm); Xv=data.matrix(val,arm)
    P=np.zeros((len(val),len(LEVELS)))
    for k,u in enumerate(LEVELS):
        yf=(fit.max_x.to_numpy()>=u).astype(int)
        yc=(cal.max_x.to_numpy()>=u).astype(int)
        if yf.min()==yf.max():
            pv=np.full(len(val),float(yf[0])); pc=np.full(len(cal),float(yf[0]))
        else:
            pc=np.zeros(len(cal)); pv=np.zeros(len(val))
            for seed in SEEDS:
                model=lgb.LGBMClassifier(random_state=seed,**LGB).fit(Xf,yf)
                pc+=model.predict_proba(Xc)[:,1]/len(SEEDS)
                pv+=model.predict_proba(Xv)[:,1]/len(SEEDS)
        if np.unique(pc).size>1 and 0<yc.sum()<len(yc):
            ir=IsotonicRegression(out_of_bounds='clip',y_min=0.,y_max=1.).fit(pc,yc)
            pv=ir.transform(pv)
        else:
            # Constant calibration target is mapped to its empirical probability.
            pv=np.full(len(val),float(yc.mean()))
        P[:,k]=pv
    P=np.minimum.accumulate(P,axis=1)
    L=np.array([np.maximum(tr.loc[tr.max_x<u,'final_x'].to_numpy()-1.3,-1.).mean()
                if (tr.max_x<u).any() else -1. for u in LEVELS])
    EV=P*(LEVELS-1.3)+(1-P)*L
    k=EV.argmax(axis=1); ev=EV[np.arange(len(val)),k]; lev=LEVELS[k]
    if not np.isfinite(P).all() or not np.isfinite(ev).all(): raise RuntimeError('Bad model output')
    return {'mint':val.mint.to_numpy(dtype=str),'day':np.array([day]*len(val)),
            'mt':val.mt.to_numpy(),'ev':ev,'level':lev,'max_x':val.max_x.to_numpy(),
            'final_x':val.final_x.to_numpy(),'fit_rows':np.array([len(fit)]),
            'cal_rows':np.array([len(cal)]),'train_max_mt':np.array([tr.mt.max()]),
            'val_min_mt':np.array([val.mt.min()]),'probability':P}

def choose_tau(history,*,available_before):
    if len(history)<4: return 1e9,[]
    table=[]
    for t in GRID:
        scores=[]; buys=[]
        for x in history[-8:]:
            resolved=x['mt']<=available_before
            tp=np.where(x['ev'][resolved]>=t,x['level'][resolved],0.)
            scores.append(float(pnl(tp,x['max_x'][resolved],x['final_x'][resolved]).sum()))
            buys.append(int((tp>0).sum()))
        mean=float(np.mean(scores)); sd=float(np.std(scores,ddof=1)); depth=float(np.mean(buys))
        table.append({'tau':float(t),'mean':mean,'sd':sd,'depth':depth,'score':mean/(sd+1)})
    band=[x for x in table if .1<=x['depth']<=2. and x['mean']>0]
    return (max(band,key=lambda x:(x['score'],x['tau']))['tau'] if band else 1e9),table

def summary(rows):
    frame=pd.DataFrame(rows)
    a=frame.pivot(index='day',columns='arm',values='pnl').sort_index()
    if set(a.columns)!={'control','reservoir'} or a.isna().any().any(): raise ValueError('Incomplete paired study')
    d=(a.reservoir-a.control).to_numpy(); n=len(d)
    rng=np.random.default_rng(20261109)
    starts=rng.integers(0,n,size=(10000,(n+1)//2))
    ix=np.stack([starts,(starts+1)%n],axis=-1).reshape(10000,-1)[:,:n]
    ci=np.quantile(d[ix].mean(axis=1),[.025,.975])
    total={c:float(a[c].sum()) for c in a}
    return {'paired_days':n,'totals':total,'abstention_total':0.,'mean_daily_delta':float(d.mean()),
            'paired_circular_block2_bootstrap_95ci':ci.tolist(),'bootstrap_draws':10000,'bootstrap_seed':20261109,
            'candidate_gate_passed':bool(n>=7 and total['reservoir']>0 and ci[0]>0),
            'control_positive':bool(total['control']>0),'post_selection_caveat':'One prespecified comparison; small calendar-day sample. Future bot scoring remains official confirmation.'}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    inp=Path(args.input); out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    seal=json.loads((inp/'INPUT_MANIFEST.json').read_text())
    for file,sha in seal['files'].items():
        if digest(inp/file)!=sha: raise RuntimeError('Input seal mismatch '+file)
    for file,sha in seal['science_source'].items():
        if digest(Path(__file__).parent/file)!=sha: raise RuntimeError('Source seal mismatch '+file)
    a=pd.read_parquet(inp/'panel.parquet');e=pd.read_parquet(inp/'edges.parquet')
    z=np.load(inp/'sequences.npz',allow_pickle=False)
    if not np.array_equal(z['mint'],a.mint.to_numpy(dtype=str)): raise RuntimeError('Sequence identity mapping differs')
    data=CausalPanel(a,e,z['T'])
    labelled=sorted(a.loc[a.max_x.notna(),'day'].unique())
    fresh=[d for d in labelled if d>='2026-10-28']
    if args.smoke: dates=['2026-10-27']
    else:
        if len(fresh)<7: raise RuntimeError('Fewer than seven prospective days')
        before=[d for d in labelled if d<fresh[0]]
        dates=before[-8:]+fresh
    t0=time.monotonic(); cache={}; rows=[]
    for day in dates:
        for arm in ('control','reservoir'):
            pred=predict_day(data,day,arm)
            path=out/(day+'_'+arm+'.npz');atomic_npz(path,**pred)
            history=[cache[(d,arm)] for d in dates if d<day and (d,arm) in cache]
            tau,grid=choose_tau(history,available_before=float(pred['val_min_mt'][0])-WIN-EMBARGO)
            cache[(day,arm)]=pred
            if day in fresh or args.smoke:
                tp=np.where(pred['ev']>=tau,pred['level'],0.)
                row={'day':day,'arm':arm,'tau':tau,'pnl':float(pnl(tp,pred['max_x'],pred['final_x']).sum()),
                     'buys':int((tp>0).sum()),'tokens':len(tp),'elapsed_seconds':time.monotonic()-t0}
                rows.append(row);atomic_json(out/'PROGRESS.json',rows)
                atomic_json(out/(day+'_'+arm+'_tau.json'),grid)
                print(json.dumps(row),flush=True)
            if time.monotonic()-t0>5300: raise RuntimeError('Bounded study time exhausted; preserve completed folds')
    result={'smoke_only':True,'elapsed_seconds':time.monotonic()-t0,'rows':rows} if args.smoke else summary(rows)
    result['elapsed_seconds']=time.monotonic()-t0
    result['input_manifest_sha256']=digest(inp/'INPUT_MANIFEST.json')
    result['source_sha256']={x:digest(Path(__file__).parent/x) for x in seal['science_source']}
    atomic_json(out/'SUMMARY.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
