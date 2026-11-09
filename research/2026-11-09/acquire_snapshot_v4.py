"""Rebuild all current inputs after the historical cache failed raw-data parity.

IO-only correction; the preregistered science/protocol remain byte-identical.
Every raw file and completed derived day has a verified recovery checkpoint.
"""
import hashlib,json,os,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import numpy as np,pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'science'))
from legacy_features import curve_feats,amm_prep,amm_feats,amm_seq
CURVE=['mint','ts_rel','direction','failed','sol','new_price','wallet','fee_payer','slot']
AMM=['mint','ts_rel','direction','sol','new_price','signing_wallet','fee_payer','slot','base_token',
     'pool_base_token_reserves_after','pool_quote_token_reserves_after','fee']
CRE=['bundle_size','gas_used','amount_of_instructions','amount_of_lookup_reads','amount_of_lookup_writes',
     'bundled_buys','bundled_buys_count','dev_balance','creation_ix_index','pf_program_index',
     'direct_pf_invocation','version','mayhem_mode','is_cashback_enabled']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)

def main():
 accepted=json.loads((R/'SCIENCE_REVIEW_ACCEPTED.json').read_text())
 prereg=json.loads((R/'PROSPECTIVE_PREREGISTRATION.json').read_text())
 if accepted['source_hashes']!=prereg['science_reviewed_hashes']:raise RuntimeError('Seal mismatch')
 for f,h in accepted['source_hashes'].items():
  if sha(R/f)!=h:raise RuntimeError('Scientific source changed')
 meta=json.loads((R/'CURRENT_FILE_METADATA_ALL.json').read_text())['response']['files']
 sizes={x['name']:x['totalBytes'] for x in meta}
 root=R/'fresh-snapshot';raw=root/'raw';derived=R/'fresh-derived-v4';derived.mkdir(exist_ok=True)
 output=R/'fresh-rebuilt-input';output.mkdir(exist_ok=True)
 if (output/'INPUT_MANIFEST.json').exists():raise RuntimeError('Snapshot complete; do not run twice')
 amendment=R/'ACQUISITION_AMENDMENT_03.json';this_sha=sha(__file__)
 if amendment.exists():
  v=json.loads(amendment.read_text())
  if v['acquisition_v4_sha256']!=this_sha or v['preregistration_sha256']!=sha(R/'PROSPECTIVE_PREREGISTRATION.json'):raise RuntimeError('Recovery source or seal changed')
 else:
  save(amendment,{'recorded_utc':datetime.now(timezone.utc).isoformat(),
   'scope':'Rebuild every current raw date; historic cache no longer used in study inputs',
   'cause':'One of378 common cached tokens differs on10curve fields; original and new formulas match current raw exactly, but old raw/version is unavailable to identify the cause',
   'acquisition_v4_sha256':this_sha,'preregistration_sha256':sha(R/'PROSPECTIVE_PREREGISTRATION.json'),
   'scientific_hashes_unchanged':accepted['source_hashes'],'prospective_predictive_metric_seen':False,
   'caveat':'Newly acquired labelled dates are historical, not live future forecasts; only bot December rotations are official forward confirmation.'})
 downloads=json.loads((root/'DOWNLOADS.json').read_text())
 api=KaggleApi();api.authenticate()
 def fetch(name):
  if name not in sizes or Path(name).name!=name:raise ValueError('Unexpected organiser filename')
  p=raw/name
  if p.exists():
   old=downloads.get(name)
   if old is None or p.stat().st_size!=sizes[name] or sha(p)!=old['sha256']:raise RuntimeError('Unverified existing raw '+name)
   return p
  print(json.dumps({'download_start':name,'bytes':sizes[name]}),flush=True)
  try:api.competition_download_file('memecoins-pump-or-dump',name,path=str(raw),quiet=True)
  except Exception as exc:raise RuntimeError('Download failed: '+type(exc).__name__) from None
  if not p.exists():
   with zipfile.ZipFile(raw/(name+'.zip')) as z:
    if z.namelist()!=[name]:raise ValueError('Unexpected ZIP contents')
    with z.open(name) as src,p.open('xb') as dst:
     import shutil
     shutil.copyfileobj(src,dst)
  if p.stat().st_size!=sizes[name]:raise ValueError('Provider metadata changed '+name)
  downloads[name]={'bytes':p.stat().st_size,'sha256':sha(p)};save(root/'DOWNLOADS.json',downloads)
  return p
 paths={f:fetch(f) for f in ('train_archive.csv','train.csv','test.csv','sample_submission.csv','pumpfun_token_creation.csv')}
 tr=pd.concat([pd.read_csv(paths[f]) for f in ('train_archive.csv','train.csv')],ignore_index=True).rename(columns={'finalx':'final_x'})
 if tr.mint.duplicated().any() or not np.isfinite(tr[['mt','max_x','final_x']]).all().all():raise ValueError('Bad training labels')
 te=pd.read_csv(paths['test.csv']);sample=pd.read_csv(paths['sample_submission.csv'])
 if set(tr.mint)&set(te.mint) or te.mint.duplicated().any() or set(te.mint)!=set(sample.mint):raise ValueError('Bad train/test identities')
 te['max_x']=np.nan;te['final_x']=np.nan
 labels=pd.concat([tr,te],ignore_index=True)
 labels['day']=pd.to_datetime(labels.mt,unit='s').dt.strftime('%Y-%m-%d')
 labels=labels.sort_values('mt',kind='stable').reset_index(drop=True)
 fresh=sorted(labels.loc[labels.max_x.notna()&(labels.day>='2026-10-28'),'day'].unique())
 if len(fresh)<7:raise RuntimeError('Too few prespecified held-out dates')
 creation=pd.read_csv(paths['pumpfun_token_creation.csv']).drop_duplicates('mint')
 c=creation[['mint','created_ts','creator']].copy()
 for k in CRE:
  if k not in creation:raise ValueError('Creation field missing '+k)
  c['x_'+k]=pd.to_numeric(creation[k],errors='coerce')
 c['x_token2022']=(~creation.token_program.fillna('').astype(str).str.startswith('Tokenkeg')).astype(int)
 c['x_bundle_struct_n']=creation.bundle_structure.fillna('').astype(str).str.count(',')+1
 c['x_symbol_len']=creation.symbol.fillna('').astype(str).str.len();c['x_name_len']=creation.name.fillna('').astype(str).str.len()
 feature_parts=[];edge_parts=[];seq_parts=[];mint_parts=[];day_seals={}
 dates=sorted(labels.day.unique())
 for day in dates:
  d=derived/day;d.mkdir(exist_ok=True);checkpoint=d/'DAY_MANIFEST.json'
  if checkpoint.exists():
   seal=json.loads(checkpoint.read_text())
   if seal['source_sha256']!=this_sha:raise RuntimeError('Derived day from different source')
   for f,h in seal['files'].items():
    if sha(d/f)!=h:raise RuntimeError('Derived day cache mismatch')
  else:
   cp=fetch('day_'+day+'_pumpfun_all_swaps.parquet');ap=fetch('day_'+day+'_pumpswap_all_swaps.parquet')
   curve=pd.read_parquet(cp,columns=CURVE)
   if not (curve.ts_rel<=60).all():raise ValueError('Curve crosses visible decision')
   amm=amm_prep(pd.read_parquet(ap,columns=AMM))
   if not amm.ts_rel.between(0,60,inclusive='both').all():raise ValueError('AMM crosses visible decision')
   F=curve_feats(curve).join(amm_feats(amm),how='outer');F.index.name='mint'
   F.reset_index().to_parquet(d/'features.parquet',index=False)
   m,T=amm_seq(amm);np.savez_compressed(d/'sequences.npz',mint=np.asarray(m,dtype=str),T=T)
   b=amm[amm.is_buy];E=b.groupby(['mint','wallet'],as_index=False).sol.sum()
   E.to_parquet(d/'edges.parquet',index=False)
   seal={'source_sha256':this_sha,'raw_hashes':{p.name:downloads[p.name]['sha256'] for p in (cp,ap)},
         'files':{f:sha(d/f) for f in ('features.parquet','sequences.npz','edges.parquet')}}
   save(checkpoint,seal);del curve,amm,F,T,E
  feature_parts.append(pd.read_parquet(d/'features.parquet'))
  edge_parts.append(pd.read_parquet(d/'edges.parquet'))
  z=np.load(d/'sequences.npz',allow_pickle=False);mint_parts.append(z['mint']);seq_parts.append(z['T'])
  day_seals[day]=sha(checkpoint)
  save(derived/'PROGRESS.json',{'completed_dates':list(day_seals),'total_dates':len(dates),'prospective_predictive_metric_seen':False})
  print(json.dumps({'derived_day':day,'complete':len(day_seals),'total':len(dates),'prospective_metric_computed':False}),flush=True)
 F=pd.concat(feature_parts,ignore_index=True)
 if F.mint.duplicated().any():raise ValueError('Mint appears in several raw dates')
 panel=labels.merge(F,on='mint',how='left',validate='one_to_one').merge(c,on='mint',how='left',validate='one_to_one')
 if not panel.mint.equals(labels.mint) or panel.c_n.isna().any() or panel.creator.isna().any():raise ValueError('Incomplete current raw panel')
 panel['x_age_s']=panel.mt-panel.created_ts;dt=pd.to_datetime(panel.mt,unit='s')
 panel['x_hour']=dt.dt.hour;panel['x_dow']=dt.dt.day_of_week
 allm=np.concatenate(mint_parts);allT=np.concatenate(seq_parts)
 if len(np.unique(allm))!=len(allm):raise ValueError('Duplicate sequence identity')
 ix=pd.Index(allm).get_indexer(panel.mint)
 if (ix<0).any():raise ValueError('Missing sequence identity')
 panel.to_parquet(output/'panel.parquet',index=False)
 np.savez_compressed(output/'sequences.npz',mint=panel.mint.to_numpy(dtype=str),T=allT[ix])
 E=pd.concat(edge_parts,ignore_index=True)
 if E.duplicated(['mint','wallet']).any():raise ValueError('Duplicate current wallet edge')
 E.to_parquet(output/'edges.parquet',index=False)
 manifest={'files':{f:sha(output/f) for f in ('panel.parquet','sequences.npz','edges.parquet')},
   'science_source':{f:sha(R/'science'/f) for f in ('study.py','legacy_features.py')},
   'protocol_sha256':sha(R/'PROTOCOL.md'),'preregistration_sha256':sha(R/'PROSPECTIVE_PREREGISTRATION.json'),
   'acquisition_source_sha256':this_sha,'derived_day_manifests':day_seals,'raw_downloads_sha256':sha(root/'DOWNLOADS.json'),
   'current_raw_only':True,'historic_cache_used':False,'rows':len(panel),'train_rows':len(tr),'test_rows':len(te),
   'fresh_labelled_dates':fresh,'unlabelled_test_outcomes_present':False,'prospective_predictive_metric_seen':False}
 save(output/'INPUT_MANIFEST.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ('derived_day_manifests','files')}),flush=True)

if __name__=='__main__':main()
