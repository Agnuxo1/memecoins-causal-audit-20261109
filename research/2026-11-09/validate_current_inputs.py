"""Mandatory raw -> daily derivation -> model-input binding before any study.

Independent of the acquisition recovery branch: stale raw/derived combinations
must fail here even when their local derived-file hashes are internally valid.
"""
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow.parquet as pq

EXPECTED_ACQUISITION='e484c56d5ccf4a33feb9786c27ea540255760b0116a4aefbff12377df2b069b5'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def verify_raw(raw,hashes,downloads,sizes,expected_names):
 if set(hashes)!=set(expected_names):raise ValueError('Raw members differ from required snapshot files')
 for name,value in hashes.items():
  if Path(name).name!=name:raise ValueError('Unsafe raw member')
  p=Path(raw)/name;record=downloads.get(name)
  if record is None or record['sha256']!=value or record['bytes']!=sizes[name]:raise ValueError('Raw download binding differs')
  if p.stat().st_size!=sizes[name] or sha(p)!=value:raise ValueError('Current raw file differs from derived-day seal')

def row_hash_update(h,frame):
 h.update(pd.util.hash_pandas_object(frame[['mint','wallet','sol']],index=False).to_numpy(dtype='<u8').tobytes())

def main():
 r=Path(__file__).resolve().parent;inp=r/'fresh-rebuilt-input';raw=r/'fresh-snapshot/raw';derived=r/'fresh-derived-v4'
 receipt=inp/'INPUT_VALIDATION_RECEIPT.json'
 if receipt.exists():raise RuntimeError('Validation already recorded; inspect receipt rather than repeat')
 m=json.loads((inp/'INPUT_MANIFEST.json').read_text())
 a=json.loads((r/'SCIENCE_REVIEW_ACCEPTED.json').read_text());p=json.loads((r/'PROSPECTIVE_PREREGISTRATION.json').read_text())
 if m['acquisition_source_sha256']!=EXPECTED_ACQUISITION or sha(r/'acquire_snapshot_v4.py')!=EXPECTED_ACQUISITION:raise ValueError('Acquisition source differs')
 if a['source_hashes']!=p['science_reviewed_hashes']:raise ValueError('Scientific review/seal differs')
 for f,h in a['source_hashes'].items():
  if sha(r/f)!=h:raise ValueError('Reviewed scientific source changed')
 if m['protocol_sha256']!=sha(r/'PROTOCOL.md') or m['preregistration_sha256']!=sha(r/'PROSPECTIVE_PREREGISTRATION.json'):raise ValueError('Protocol/preregistration differs')
 if set(m['files'])!={'panel.parquet','sequences.npz','edges.parquet'}:raise ValueError('Unexpected input members')
 for f,h in m['files'].items():
  if sha(inp/f)!=h:raise ValueError('Assembled input hash differs')
 downloads=json.loads((r/'fresh-snapshot/DOWNLOADS.json').read_text())
 if sha(r/'fresh-snapshot/DOWNLOADS.json')!=m['raw_downloads_sha256']:raise ValueError('Raw snapshot manifest differs')
 if sha(r/'CURRENT_FILE_METADATA_ALL.json')!=p['metadata_sha256']:raise ValueError('Provider metadata differs from preregistered snapshot')
 meta=json.loads((r/'CURRENT_FILE_METADATA_ALL.json').read_text())['response']['files'];sizes={x['name']:x['totalBytes'] for x in meta}
 csvs=['train_archive.csv','train.csv','test.csv','sample_submission.csv','pumpfun_token_creation.csv']
 verify_raw(raw,{f:downloads[f]['sha256'] for f in csvs},downloads,sizes,csvs)
 panel=pd.read_parquet(inp/'panel.parquet');z=np.load(inp/'sequences.npz',allow_pickle=False);T=z['T'];mints=z['mint']
 if not np.array_equal(mints,panel.mint.to_numpy(dtype=str)) or T.shape!=(len(panel),20,7) or not np.isfinite(T).all():raise ValueError('Assembled sequence identity/shape/values differ')
 train=pd.concat([pd.read_csv(raw/f) for f in ('train_archive.csv','train.csv')],ignore_index=True).rename(columns={'finalx':'final_x'})
 test=pd.read_csv(raw/'test.csv');test['max_x']=np.nan;test['final_x']=np.nan
 official=pd.concat([train,test],ignore_index=True).set_index('mint')
 if official.index.duplicated().any() or set(official.index)!=set(panel.mint):raise ValueError('Official input identities differ')
 pd.testing.assert_frame_equal(panel.set_index('mint')[['mt','max_x','final_x']],official.reindex(panel.mint)[['mt','max_x','final_x']],check_dtype=False,check_exact=True)
 if not panel.day.equals(pd.to_datetime(panel.mt,unit='s').dt.strftime('%Y-%m-%d')):raise ValueError('Calendar mapping differs')
 dates=sorted(panel.day.unique())
 if set(dates)!=set(m['derived_day_manifests']):raise ValueError('Derived dates incomplete')
 feature_cols=None;expected_edges=hashlib.sha256();edge_rows=0
 for day in dates:
  d=derived/day;seal_path=d/'DAY_MANIFEST.json'
  if sha(seal_path)!=m['derived_day_manifests'][day]:raise ValueError('Derived day manifest differs')
  seal=json.loads(seal_path.read_text())
  if seal['source_sha256']!=EXPECTED_ACQUISITION or set(seal['files'])!={'features.parquet','sequences.npz','edges.parquet'}:raise ValueError('Derived source/members differ')
  for f,h in seal['files'].items():
   if sha(d/f)!=h:raise ValueError('Derived file hash differs')
  names=['day_'+day+'_'+t+'_all_swaps.parquet' for t in ('pumpfun','pumpswap')]
  verify_raw(raw,seal['raw_hashes'],downloads,sizes,names)
  mask=(panel.day==day).to_numpy();part=panel.loc[mask].set_index('mint');F=pd.read_parquet(d/'features.parquet').set_index('mint')
  if F.index.duplicated().any():raise ValueError('Duplicated feature identity')
  if feature_cols is None:feature_cols=list(F.columns)
  if list(F.columns)!=feature_cols:raise ValueError('Daily feature schema changed')
  pd.testing.assert_frame_equal(part[feature_cols],F.reindex(part.index),check_dtype=False,check_exact=True)
  q=np.load(d/'sequences.npz',allow_pickle=False);idx=pd.Index(q['mint']).get_indexer(part.index)
  if (idx<0).any() or not np.array_equal(q['T'][idx],T[mask]):raise ValueError('Daily/assembled sequence mapping differs')
  E=pd.read_parquet(d/'edges.parquet');row_hash_update(expected_edges,E);edge_rows+=len(E)
 actual_edges=hashlib.sha256();seen=0
 for batch in pq.ParquetFile(inp/'edges.parquet').iter_batches(batch_size=200000,columns=['mint','wallet','sol']):
  frame=batch.to_pandas();row_hash_update(actual_edges,frame);seen+=len(frame)
 if seen!=edge_rows or actual_edges.hexdigest()!=expected_edges.hexdigest():raise ValueError('Daily/assembled wallet edge lineage differs')
 result={'observed_utc':datetime.now(timezone.utc).isoformat(),'all_passed':True,
   'raw_derived_input_binding_verified':True,'input_manifest_sha256':sha(inp/'INPUT_MANIFEST.json'),
   'validation_source_sha256':sha(__file__),'science_source':m['science_source'],
   'raw_files_checked':2*len(dates)+len(csvs),'derived_days_checked':len(dates),'panel_rows':len(panel),
   'official_label_rows_checked_exactly':len(official),'tape_feature_columns_checked_exactly':len(feature_cols),'sequence_rows_checked_exactly':len(T),
   'wallet_edge_rows_checked':edge_rows,'edge_row_fingerprint':actual_edges.hexdigest(),
   'prospective_predictive_metric_computed':False}
 receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
