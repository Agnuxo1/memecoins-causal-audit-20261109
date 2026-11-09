"""Build the same input representation from attached organiser files only.

No network, credential access, pretrained objects or external feature caches.
Francisco Angulo de Lafuente. ORCID 0009-0001-1634-7063.
"""
import gc
from pathlib import Path
import numpy as np
import pandas as pd

VISIBLE_CURVE_COLUMNS=['mint','ts_rel','direction','failed','sol','new_price','wallet','fee_payer','slot']
VISIBLE_AMM_COLUMNS=['mint','ts_rel','direction','sol','new_price','signing_wallet','fee_payer','slot','base_token',
                     'pool_base_token_reserves_after','pool_quote_token_reserves_after','fee']
CREATION_NUMERIC=['bundle_size','gas_used','amount_of_instructions','amount_of_lookup_reads','amount_of_lookup_writes',
                 'bundled_buys','bundled_buys_count','dev_balance','creation_ix_index','pf_program_index',
                 'direct_pf_invocation','version','mayhem_mode','is_cashback_enabled']

def attached_tape_path(data,day,table):
    paths=[data/('day_'+day+'_'+table+'_all_swaps.parquet'),
           data/'trades'/('day='+day)/(table+'_all_swaps.parquet')]
    for p in paths:
        if p.is_file():return p
    raise FileNotFoundError('Attached first-minute tape absent for '+day+' '+table)

def build_attached(data):
    data=Path(data)
    train_parts=[pd.read_csv(data/'train.csv')]
    if (data/'train_archive.csv').is_file():train_parts.insert(0,pd.read_csv(data/'train_archive.csv'))
    tr=pd.concat(train_parts,ignore_index=True).rename(columns={'finalx':'final_x'})
    test=pd.read_csv(data/'test.csv');sample=pd.read_csv(data/'sample_submission.csv')
    if tr.mint.duplicated().any() or test.mint.duplicated().any() or sample.mint.duplicated().any():
        raise ValueError('Duplicate organiser identity')
    if set(tr.mint)&set(test.mint) or set(sample.mint)!=set(test.mint):raise ValueError('Train/test/sample identities differ')
    if not np.isfinite(tr[['mt','max_x','final_x']]).all().all():raise ValueError('Nonfinite training target')
    test['max_x']=np.nan;test['final_x']=np.nan
    a=pd.concat([tr,test],ignore_index=True)
    a['day']=pd.to_datetime(a.mt,unit='s').dt.strftime('%Y-%m-%d')
    a=a.sort_values('mt',kind='stable').reset_index(drop=True)
    test_days=pd.to_datetime(test.mt,unit='s').dt.strftime('%Y-%m-%d').unique()
    if len(test_days)!=1:raise ValueError('Expected one test calendar day')
    creation=pd.read_csv(data/'pumpfun_token_creation.csv').drop_duplicates('mint')
    c=creation[['mint','created_ts','creator']].copy()
    for k in CREATION_NUMERIC:
        if k not in creation:raise ValueError('Creation field absent '+k)
        c['x_'+k]=pd.to_numeric(creation[k],errors='coerce')
    c['x_token2022']=(~creation.token_program.fillna('').astype(str).str.startswith('Tokenkeg')).astype(int)
    c['x_bundle_struct_n']=creation.bundle_structure.fillna('').astype(str).str.count(',')+1
    c['x_symbol_len']=creation.symbol.fillna('').astype(str).str.len()
    c['x_name_len']=creation.name.fillna('').astype(str).str.len()
    del creation
    fparts=[];eparts=[];mparts=[];tparts=[]
    for day in sorted(a.day.unique()):
        curve=pd.read_parquet(attached_tape_path(data,day,'pumpfun'),columns=VISIBLE_CURVE_COLUMNS)
        if not (curve.ts_rel<=60).all():raise ValueError('Curve exceeds visible decision time')
        amm=amm_prep(pd.read_parquet(attached_tape_path(data,day,'pumpswap'),columns=VISIBLE_AMM_COLUMNS))
        if not amm.ts_rel.between(0,60,inclusive='both').all():raise ValueError('AMM exceeds visible first minute')
        f=curve_feats(curve).join(amm_feats(amm),how='outer');f.index.name='mint'
        fparts.append(f.reset_index())
        m,t=amm_seq(amm);mparts.append(np.asarray(m,dtype=str));tparts.append(t)
        buys=amm[amm.is_buy];eparts.append(buys.groupby(['mint','wallet'],as_index=False).sol.sum())
        del curve,amm,f,m,t,buys;gc.collect()
        print('Attached data extracted:',day,flush=True)
    f=pd.concat(fparts,ignore_index=True)
    if f.mint.duplicated().any():raise ValueError('Repeated tape identity')
    panel=a.merge(f,on='mint',how='left',validate='one_to_one').merge(c,on='mint',how='left',validate='one_to_one')
    if not panel.mint.equals(a.mint) or panel.c_n.isna().any() or panel.creator.isna().any():raise ValueError('Incomplete attached panel')
    panel['x_age_s']=panel.mt-panel.created_ts;dt=pd.to_datetime(panel.mt,unit='s')
    panel['x_hour']=dt.dt.hour;panel['x_dow']=dt.dt.day_of_week
    mints=np.concatenate(mparts);tensor=np.concatenate(tparts)
    if len(np.unique(mints))!=len(mints):raise ValueError('Duplicate sequence identity')
    ix=pd.Index(mints).get_indexer(panel.mint)
    if (ix<0).any():raise ValueError('Missing sequence identity')
    tensor=tensor[ix]
    edges=pd.concat(eparts,ignore_index=True)
    if edges.duplicated(['mint','wallet']).any():raise ValueError('Duplicate wallet edge')
    return panel,edges,tensor,sample,str(test_days[0])
