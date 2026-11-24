import argparse
import importlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from pyproj import Geod

ROOT = Path(os.environ.get('GEO_CV_ROOT', '/content/train250k-cv'))
SEED = 417359
METHODS = ('catboost', 'dnn', 'siamese', 'neighbors')
ELLIPSOID = Geod(ellps='WGS84')
FLATTENING = 1 / 298.257223563
E2 = 2 * FLATTENING - FLATTENING ** 2

def get_model(method):
    if method in ('dnn', 'siamese'):
        mod = importlib.import_module('neural')
        return mod.train_predict, mod.NEURAL_CONFIGS[method]
    mod = importlib.import_module('trees' if method == 'catboost' else 'neighbors')
    return mod.train_predict, mod.CONFIGS

def decode(v):
    assert v.shape[1] == 3 and np.isfinite(v).all()
    assert np.all(np.linalg.norm(v, axis=1) > 1e-7)
    x, y, z = np.asarray(v, dtype=np.float64).T
    return np.rad2deg(np.arctan2(z, (1 - E2)*np.hypot(x,y))), np.rad2deg(np.arctan2(y,x))

def metrics(predicted, truth):
    lat, lon = decode(predicted)
    d = ELLIPSOID.inv(truth[:,1], truth[:,0], lon, lat)[2] * 0.001
    assert np.isfinite(d).all()
    mae = float(np.mean(d))
    mse = float(np.mean(np.square(d)))
    return {'mae_km': mae, 'rmse_km': mse**0.5, 'mse_km2': mse, 'n': len(d)}

def spherical_blend(a, b, b_weight):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    angle = np.arccos(np.clip(np.einsum('ij,ij->i', a, b), -1, 1))
    ordinary = (angle > 1e-4) & (angle < np.pi - 1e-4)
    w_a = np.full(len(angle), 1-b_weight)
    w_b = np.full(len(angle), b_weight)
    w_a[ordinary] = np.sin((1-b_weight)*angle[ordinary]) / np.sin(angle[ordinary])
    w_b[ordinary] = np.sin(b_weight*angle[ordinary]) / np.sin(angle[ordinary])
    result = w_a[:, None]*a + w_b[:, None]*b
    return np.where(np.linalg.norm(result, axis=1, keepdims=True) > 1e-7, result, a)

def save_json(path, obj):
    staged = path.with_suffix('.tmp')
    staged.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    staged.replace(path)

def get_folds(method, x, y, latlon, folds):
    fit, configs = get_model(method)
    for fold in range(5):
        train_idx, val_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        for position, config in enumerate(configs):
            path = ROOT / f'cv-{method}-{fold}-{position}.json'
            if path.exists():
                continue
            print(f'BEGIN {method}, fold {fold+1}/5, config {position+1}/{len(configs)}: {config}', flush=True)
            start = perf_counter()
            result = np.asarray(fit(method, config, x, y, train_idx, val_idx, SEED+fold), dtype=np.float32)
            assert result.shape == (len(val_idx), 3)
            summary = {'method': method, 'fold': fold, 'config_index': position, 'config': config,
                       'seconds': round(perf_counter()-start,2), **metrics(result, latlon[val_idx])}
            save_json(path, summary)
            np.save(ROOT / f'pred-{method}-{fold}-{position}.npy',result)
            print(f'END {method}, fold {fold+1}, config {position+1}: {summary}', flush=True)
    ranked = []
    for pos, config in enumerate(configs):
        rows = [json.loads((ROOT/f'cv-{method}-{fold}-{pos}.json').read_text()) for fold in range(5)]
        n = sum(r['n'] for r in rows)
        mae = sum(r['mae_km']*r['n'] for r in rows)/n
        mse = sum(r['mse_km2']*r['n'] for r in rows)/n
        ranked.append({'config_index':pos,'config':config,'mae_km':mae, 'rmse_km':mse**0.5,
                       'mse_km2':mse,'folds':rows,'n':n})
    if method == 'neighbors':
        dnn = json.loads((ROOT/'cv-dnn-summary.json').read_text())[0]
        for pos, config in enumerate(configs):
            for weight in (0.25, 0.50, 0.75):
                rows = []
                for fold in range(5):
                    retrieved = np.load(ROOT / f'pred-neighbors-{fold}-{pos}.npy')
                    direct = np.load(ROOT / f'pred-dnn-{fold}-{dnn["config_index"]}.npy')
                    score = metrics(spherical_blend(retrieved, direct, weight), latlon[folds == fold])
                    rows.append({'fold':fold, **score})
                n = sum(r['n'] for r in rows)
                mse = sum(r['mse_km2']*r['n'] for r in rows)/n
                mae = sum(r['mae_km']*r['n'] for r in rows)/n
                ranked.append({'config_index':pos,'config':{**config,'dnn_weight':weight},
                               'mae_km':mae,'rmse_km':mse**0.5,'mse_km2':mse,'folds':rows,'n':n})
    ranked.sort(key=lambda r:r['mse_km2'])
    save_json(ROOT/f'cv-{method}-summary.json', ranked)
    print(f'SELECT {method}: {ranked[0]["config"]}', flush=True)

def final(dev_x, dev_y):
    selections = {}
    for name in METHODS:
        path = ROOT/f'cv-{name}-summary.json'
        assert path.is_file(), f'Finish cross-validation for {name} before opening test'
        selections[name] = json.loads(path.read_text())[0]
    limit = len(dev_x)
    # Test inputs are used exclusively for prediction. Ground truth for test
    # is opened only after all four fitted models produced their predictions.
    test_x = np.load(ROOT/'test_x.npy', mmap_mode='r')
    x = np.concatenate([dev_x, test_x])
    y = np.concatenate([dev_y, np.full((len(test_x),3), np.nan, dtype=np.float32)])
    train_idx = np.arange(limit)
    test_idx = np.arange(limit,len(x))
    for name in METHODS:
        path = ROOT/f'test-{name}-pred.npy'
        if path.exists():
            continue
        fit, _ = get_model(name)
        print(f'REFIT {name} on {limit} rows: {selections[name]["config"]}', flush=True)
        run_config = dict(selections[name]['config'])
        if name != 'neighbors':
            extension = 'cbm' if name == 'catboost' else 'pt'
            run_config['save_path'] = str(ROOT/f'fitted-{name}.{extension}')
        result = np.asarray(fit(name, run_config, x, y, train_idx, test_idx, SEED+5), dtype=np.float32)
        assert result.shape == (len(test_idx),3)
        blend = selections[name]['config'].get('dnn_weight', 0)
        if blend:
            result = spherical_blend(result, np.load(ROOT/'test-dnn-pred.npy'), blend).astype(np.float32)
        decode(result)
        staged = path.with_suffix('.tmp.npy')
        np.save(staged,result)
        staged.replace(path)
    test_latlon = np.load(ROOT/'test_latlon.npy', mmap_mode='r')
    results = {name:{'selection':selections[name],'test':metrics(
        np.load(ROOT/f'test-{name}-pred.npy'),test_latlon)} for name in METHODS}
    save_json(ROOT/'results.json', {'split':json.loads((ROOT/'split.json').read_text()),'models':results})
    print(json.dumps({name:r['test'] for name,r in results.items()}, indent=2),flush=True)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('phase', choices=(*METHODS,'final'))
    phase = args.parse_args().phase
    x = np.load(ROOT / 'dev_x.npy',mmap_mode='r')
    y = np.load(ROOT / 'dev_y.npy',mmap_mode='r')
    if phase == 'final':
        # Resume completed five-fold trials and add the deeper, longer CatBoost trials
        # before loading held-out inputs or scores.
        get_folds('catboost',x,y,np.load(ROOT/'dev_latlon.npy'),np.load(ROOT/'folds.npy'))
        final(x,y)
    else:
        get_folds(phase,x,y,np.load(ROOT/'dev_latlon.npy'),np.load(ROOT/'folds.npy'))

if __name__=='__main__':
    main()
