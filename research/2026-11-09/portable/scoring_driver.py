"""Use the exact reviewed model/threshold functions to score one unlabelled day."""
import json
import os
import time
from pathlib import Path

def write_scoring_submission(pred,tau,sample,output):
    tp=np.where(pred['ev']>=tau,pred['level'],0.).astype(float)
    if len(set(pred['mint']))!=len(pred['mint']):raise ValueError('Duplicate prediction identity')
    mapping=pd.Series(tp,index=pred['mint'])
    sub=sample[['mint']].copy();sub['tp']=sub.mint.map(mapping)
    if len(sub)!=len(sample) or not sub.mint.equals(sample.mint):raise ValueError('Submission order/length differs')
    if sub.tp.isna().any() or not np.isfinite(sub.tp).all() or not ((sub.tp==0)|sub.tp.between(3,100,inclusive='both')).all():
        raise ValueError('Invalid or missing legal prediction')
    output=Path(output);tmp=output.with_suffix(output.suffix+'.tmp')
    sub.to_csv(tmp,index=False);os.replace(tmp,output)
    return sub

def score_attached(data,arm,output='submission.csv'):
    if arm not in ('control','reservoir'):raise ValueError('Unknown fixed arm')
    started=time.monotonic()
    a,e,T,sample,test_day=build_attached(data)
    panel=CausalPanel(a,e,T)
    days=sorted(a.loc[a.max_x.notna()&(a.day<test_day),'day'].unique())[-8:]
    if len(days)<4:raise ValueError('Insufficient threshold history')
    history=[predict_day(panel,day,arm)for day in days]
    current=predict_day(panel,test_day,arm)
    cutoff=float(current['val_min_mt'][0])-WIN-EMBARGO
    tau,_=choose_tau(history,available_before=cutoff)
    sub=write_scoring_submission(current,tau,sample,output)
    print(json.dumps({'test_day':test_day,'arm':arm,'rows':len(sub),'buys':int((sub.tp>0).sum()),
                     'tau':tau,'elapsed_seconds':time.monotonic()-started,'trained_inside_this_run':True,
                     'network_used':False,'gpu_used':False}),flush=True)
    return sub
