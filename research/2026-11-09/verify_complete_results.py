"""Independent deterministic verification of the one complete frozen comparison.

Does not fit, retune, access a partial run, publish data, or make a submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEVELS = np.array([3., 4., 5., 7., 10., 15., 20.])

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def payoff(tp, high, final):
    # Official idealised per-token score, no real transaction.
    out = np.zeros(len(tp), dtype=float)
    buy = tp > 0
    hit = buy & (high >= tp)
    miss = buy & ~hit
    out[hit] = tp[hit] - 1.3
    out[miss] = np.maximum(final[miss] - 1.3, -1.)
    return out

def threshold(history, cutoff):
    if len(history) < 4:
        return 1e9, []
    grid = []
    for tau in np.round(np.arange(0., 2.51, .1), 2):
        days, counts = [], []
        for old in history[-8:]:
            mask = old['mt'] <= cutoff
            tp = np.where(old['ev'][mask] >= tau, old['level'][mask], 0.)
            days.append(float(payoff(tp, old['max_x'][mask], old['final_x'][mask]).sum()))
            counts.append(int(np.count_nonzero(tp)))
        avg, std, depth = float(np.mean(days)), float(np.std(days, ddof=1)), float(np.mean(counts))
        grid.append({'tau': float(tau), 'mean': avg, 'sd': std, 'depth': depth, 'score': avg/(std+1.)})
    legal = [r for r in grid if .1 <= r['depth'] <= 2. and r['mean'] > 0.]
    legal.sort(key=lambda r: (r['score'], r['tau']))
    return (legal[-1]['tau'] if legal else 1e9), grid

def equal_numeric(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('export', type=Path)
    args = parser.parse_args()
    assert not args.export.is_symlink(), 'Unexpected export symlink'
    root = args.export.resolve()
    # Admission must precede reads/hashes of any metric/prediction payload.
    status_path = root/'RUN_STATUS.json'
    assert not status_path.is_symlink()
    run = json.loads(status_path.read_text())
    assert run['state'] == 'succeeded' and run['exit_code'] == 0 and run['gpus_requested'] == 0
    assert run['cuda_visible_devices'] == '' and run['threads'] == 4
    fixed_dates = pd.date_range('2026-10-20','2026-11-07').strftime('%Y-%m-%d').tolist()
    fixed_fresh = [d for d in fixed_dates if d >= '2026-10-28']
    expected_npz = {f'{d}_{arm}.npz' for d in fixed_dates for arm in ('control','reservoir')}
    assert {p.name for p in (root/'results').glob('*.npz')} == expected_npz
    expected_tau = {f'{d}_{arm}_tau.json' for d in fixed_fresh for arm in ('control','reservoir')}
    assert {p.name for p in (root/'results').glob('*_tau.json')} == expected_tau
    required = {'RUN_STATUS.json','results/SUMMARY.json','results/PROGRESS.json'}
    required |= {'results/'+name for name in expected_npz | expected_tau}
    all_paths = list(root.rglob('*'))
    assert not any(p.is_symlink() for p in all_paths), 'Unexpected export member symlink'
    actual_files = {p.relative_to(root).as_posix() for p in all_paths if p.is_file() and p.name != 'EXPORT_MANIFEST.json'}
    assert required <= actual_files, 'Missing complete-result members'
    receipt = json.loads((root/'EXPORT_MANIFEST.json').read_text())
    assert set(receipt['files']) == actual_files, 'Manifest must cover every exported payload file'
    for name, item in receipt['files'].items():
        path = (root/name).resolve()
        assert path.is_relative_to(root) and path.is_file()
        assert sha(path) == item['sha256'] and path.stat().st_size == item['bytes']
    result = json.loads((root/'results/SUMMARY.json').read_text())
    inp = ROOT/'fresh-rebuilt-input'
    manifest = json.loads((inp/'INPUT_MANIFEST.json').read_text())
    assert sha(inp/'INPUT_MANIFEST.json') == run['input_manifest_sha256'] == result['input_manifest_sha256']
    assert manifest['science_source'] == run['science_source'] == result['source_sha256']
    for name, expected in manifest['science_source'].items():
        assert sha(ROOT/'science'/name) == expected
    for name, expected in manifest['files'].items():
        assert sha(inp/name) == expected
    panel = pd.read_parquet(inp/'panel.parquet', columns=['mint','day','mt','max_x','final_x'])
    all_days = sorted(panel.loc[panel.max_x.notna(),'day'].unique())
    fresh = [d for d in all_days if d >= '2026-10-28']
    dates = [d for d in all_days if d < fresh[0]][-8:] + fresh
    assert len(dates) == 19 and len(fresh) == 11
    expected_names = {f'{d}_{arm}.npz' for d in dates for arm in ('control','reservoir')}
    assert {p.name for p in (root/'results').glob('*.npz')} == expected_names
    reported = json.loads((root/'results/PROGRESS.json').read_text())
    assert len(reported) == 22
    rows, histories = [], {'control': [], 'reservoir': []}
    for day in dates:
        truth = panel[panel.day == day]
        cutoff = float(truth.mt.min()) - 7200.
        train = panel[panel.max_x.notna() & (panel.mt <= cutoff)]
        cal_days = sorted(train.day.unique())[-5:]
        # Exactly the training identities in the five calibration days.
        cal = train[train.day.isin(cal_days)]
        fit = train[(train.day < cal_days[0]) & (train.mt <= float(cal.mt.min())-7200.)]
        for arm in ('control','reservoir'):
            with np.load(root/'results'/f'{day}_{arm}.npz', allow_pickle=False) as archive:
                pred = {k: archive[k] for k in archive.files}
            np.testing.assert_array_equal(pred['mint'], truth.mint.to_numpy(dtype=str))
            np.testing.assert_array_equal(pred['day'], np.full(len(truth), day))
            for key in ('mt','max_x','final_x'):
                equal_numeric(pred[key], truth[key].to_numpy())
            assert int(pred['fit_rows'][0]) == len(fit) and int(pred['cal_rows'][0]) == len(cal)
            equal_numeric(pred['train_max_mt'], [train.mt.max()])
            equal_numeric(pred['val_min_mt'], [truth.mt.min()])
            assert float(pred['train_max_mt'][0]) <= cutoff
            probability = pred['probability']
            assert probability.shape == (len(truth),7)
            assert np.isfinite(probability).all() and ((probability >= 0.) & (probability <= 1.)).all()
            assert (np.diff(probability, axis=1) <= 1e-14).all()
            loss = np.array([np.maximum(train.loc[train.max_x < u,'final_x'].to_numpy()-1.3,-1.).mean()
                             if (train.max_x < u).any() else -1. for u in LEVELS])
            ev = probability*(LEVELS-1.3)+(1.-probability)*loss
            selected = ev.argmax(axis=1)
            equal_numeric(pred['level'], LEVELS[selected])
            equal_numeric(pred['ev'], ev[np.arange(len(truth)),selected])
            tau, grid = threshold(histories[arm], cutoff)
            histories[arm].append(pred)
            if day in fresh:
                tp = np.where(pred['ev'] >= tau, pred['level'], 0.)
                row = {'day': day, 'arm': arm, 'tau': tau,
                       'pnl': float(payoff(tp,pred['max_x'],pred['final_x']).sum()),
                       'buys': int(np.count_nonzero(tp)), 'tokens': len(tp)}
                actual = next(x for x in reported if x['day']==day and x['arm']==arm)
                assert row.keys() <= actual.keys()
                for key in row:
                    if isinstance(row[key],str): assert row[key] == actual[key]
                    else: equal_numeric(row[key], actual[key])
                actual_grid = json.loads((root/'results'/f'{day}_{arm}_tau.json').read_text())
                assert len(actual_grid) == len(grid)
                for expected, observed in zip(grid,actual_grid):
                    assert expected.keys() == observed.keys()
                    equal_numeric(list(expected.values()), list(observed.values()))
                rows.append(row)
    days = pd.DataFrame(rows).pivot(index='day',columns='arm',values='pnl').sort_index()
    delta = (days['reservoir']-days['control']).to_numpy()
    n = len(days)
    rng = np.random.default_rng(20261109)
    block_start = rng.integers(n,size=(10000,(n+1)//2))
    draws = np.empty((10000,2*block_start.shape[1]),dtype=int)
    draws[:,0::2] = block_start
    draws[:,1::2] = (block_start+1)%n
    ci = np.quantile(np.mean(delta[draws[:,:n]],axis=1), [.025,.975])
    totals = {arm:float(days[arm].sum()) for arm in ('control','reservoir')}
    passed = bool(n >= 7 and totals['reservoir'] > 0. and ci[0] > 0.)
    assert result['paired_days'] == n == 11
    assert result['bootstrap_draws'] == 10000 and result['bootstrap_seed'] == 20261109
    for arm in totals: equal_numeric(totals[arm],result['totals'][arm])
    equal_numeric(ci,result['paired_circular_block2_bootstrap_95ci'])
    equal_numeric(delta.mean(),result['mean_daily_delta'])
    assert passed == result['candidate_gate_passed']
    assert result['control_positive'] == (totals['control'] > 0.)
    out = {'all_checks_passed':True, 'export_manifest_sha256':sha(root/'EXPORT_MANIFEST.json'),
           'verification_source_sha256':sha(__file__), 'prediction_artifacts':38,
           'paired_heldout_days':n,'rows':rows,'totals':totals,
           'mean_daily_candidate_minus_control':float(delta.mean()),
           'paired_block2_95ci':ci.tolist(),'candidate_gate_passed':passed,
           'not_official_score':True,'no_further_trials_performed':True}
    target = ROOT/'COMPLETE_RESULT_VERIFICATION.json'
    assert not target.exists(), 'Verification already exists; inspect rather than overwrite'
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__ == '__main__':
    main()
