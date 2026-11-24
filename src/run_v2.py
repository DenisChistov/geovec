"""Compare train-only model variants, lock the winner, then score the fixed test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import KFold
import torch

from experiment import metrics, save_json, spherical_blend
from retrieval_v2 import retrieve

DATA = Path(os.environ.get('GEO_CV_ROOT', '/content/train250k-cv'))
OUT = DATA / 'v2'
SEED = 523748
REFERENCE = {'description': 'previous experiment on the same fixed holdout', 'mae_km': 448.194, 'rmse_km': 1122.631}
NAMES = ('joint', 'short', 'broad', 'geo', 'street', 'favor_geo')
CELL_COUNTS = (512, 1024)


def open_dev():
    x = np.load(DATA / 'dev_x.npy',mmap_mode='r')
    y = np.load(DATA / 'dev_y.npy',mmap_mode='r')
    truth = np.load(DATA / 'dev_latlon.npy',mmap_mode='r')
    folds = np.load(DATA / 'folds.npy')
    if x.shape != (212_500,1280) or np.bincount(folds).tolist() != [42_500] * 5:
        raise ValueError('The fixed 85% and five folds are required.')
    return x,y,truth,folds


def split_fingerprint():
    digest = hashlib.sha256()
    for name in ('dev_rows.npy', 'test_rows.npy', 'folds.npy'):
        digest.update((DATA/name).read_bytes())
    return digest.hexdigest()


def atomic_arrays(path, **arrays):
    staging = path.with_suffix('.tmp.npz')
    np.savez(staging, **arrays)
    staging.replace(path)


def cached_retrieval(name, X, Y, train, query):
    path = OUT / f'retrieval-{name}.npz'
    if path.is_file():
        with np.load(path) as file:
            return dict(file)
    result = retrieve(X,Y,train,query)
    atomic_arrays(path,**result)
    return result


def make_sections(folds):
    for fold in range(5):
        train, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        inner = list(KFold(3, shuffle=True, random_state=SEED+fold).split(train))
        yield fold,train,val,inner


def build_retrieval(X, Y, folds):
    for fold, train, val, inner in make_sections(folds):
        t = perf_counter()
        cached_retrieval(f'o{fold}', X,Y,train,val)
        for i,(subtrain, subquery) in enumerate(inner):
            cached_retrieval(f'o{fold}-i{i}', X,Y,train[subtrain],train[subquery])
        print(f'Retrieval fold {fold+1}/5: {perf_counter()-t:.1f} s',flush=True)


def unit(x):
    return x / np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-9)


def baseline(dnn, ret):
    return spherical_blend(ret['joint'],dnn,.5).astype(np.float32)


def geo_adaptive(dnn, ret):
    sim = ret['statistics'][:,0]
    scatter = ret['statistics'][:,3]
    model_agreement = ret['statistics'][:,-3]
    # Similar and geographically coherent references can get more weight.
    confidence = (0.75*sim + 0.25*model_agreement - 0.65*scatter)
    dnn_weight = np.clip(1.08 - .90*confidence, .18, .74)
    # Use the short estimate only when the top references are very coherent.
    short_weight = np.clip((confidence - .55)*2,.0,.70)
    neighbor = unit(ret['joint'] * (1-short_weight[:,None]) + ret['short'] * short_weight[:,None])
    return unit(neighbor * (1-dnn_weight[:,None]) + dnn*dnn_weight[:,None])


def features(dnn, ret, reduced):
    estimates = [dnn] + [ret[name] for name in NAMES]
    agreements = [(a*b).sum(axis=1) for i,a in enumerate(estimates) for b in estimates[i+1:]]
    return np.column_stack([*estimates,ret['statistics'],*agreements,reduced]).astype(np.float32)


@torch.inference_mode()
def pca_features(X, train, query, seed):
    # Fit and center within train only; sample rows solely from this part.
    rng = np.random.default_rng(seed)
    sample = rng.choice(train, size=min(45_000,len(train)), replace=False)
    xt = torch.as_tensor(X[sample], device='cuda')
    center = xt.mean(dim=0)
    torch.manual_seed(seed)
    _,_, projection = torch.pca_lowrank(xt-center, q=36, center=False, niter=2)
    projection = projection[:,:32].contiguous()
    train_feat = ((torch.as_tensor(X[train],device='cuda')-center) @ projection).cpu().numpy()
    query_feat = ((torch.as_tensor(X[query],device='cuda')-center) @ projection).cpu().numpy()
    return train_feat,query_feat,center.cpu().numpy(),projection.cpu().numpy()


def checked_prediction(pred):
    pred=np.asarray(pred,dtype=np.float32)
    if not np.isfinite(pred).all() or (np.linalg.norm(pred,axis=1)<.5).any():
        raise ValueError('Invalid directions')
    return unit(pred)


def old_direct(X,Y,train,query,seed,save_path=None):
    from neural import train_predict, NEURAL_CONFIGS
    config = dict(NEURAL_CONFIGS['dnn'][1])
    if save_path:
        config['save_path']=str(save_path)
    return train_predict('dnn',config,X,Y,train,query,seed)


def deep_prediction(name, X,Y,train,query,seed, save_path=None):
    if name == 'v1_dnn':
        return old_direct(X,Y,train,query,seed,save_path)
    from deep_v2 import CONFIGS,fit_predict
    return fit_predict(X,Y,train,query,CONFIGS[name],seed,save_path=save_path)


def deep_cache(key, name, X,Y,train,query,seed):
    path=OUT / f'deep-{key}-{name}.npz'
    if path.exists():
        with np.load(path) as cache:
            return cache['xyz']
    pred=checked_prediction(deep_prediction(name,X,Y,train,query,seed))
    atomic_arrays(path,xyz=pred)
    return pred


def config_registry():
    from deep_v2 import CONFIGS as deeps
    from stack_v2 import CONFIGS as stacks
    meta_base=next(iter(deeps))
    return deeps,stacks,meta_base


def collect_fold(fold,X,Y,truth,train,val,inner):
    from stack_v2 import fit_predict
    deeps,stacks,meta = config_registry()
    summary=OUT / f'fold-{fold}.json'
    if summary.exists():
        return json.loads(summary.read_text())
    begin=perf_counter()
    ret=cached_retrieval(f'o{fold}',X,Y,train,val)
    outputs={}
    old=deep_cache(f'o{fold}','v1_dnn',X,Y,train,val,417359+fold)
    outputs['reference_v1'] = baseline(old,ret)
    outputs['retrieval'] = ret['joint']
    outputs['retrieval_geo_bias'] = ret['favor_geo']
    for name in deeps:
        xyz=deep_cache(f'o{fold}',name,X,Y,train,val,SEED+fold)
        outputs[f'direct_{name}'] = xyz
        outputs[f'hybrid_{name}'] = baseline(xyz,ret)
    dnn=outputs[f'direct_{meta}']
    outputs['hybrid_confidence'] = geo_adaptive(dnn,ret)

    proj_train,proj_val,*_=pca_features(X,train,val,SEED+fold)
    meta_val=features(dnn,ret,proj_val)
    meta_train=np.empty((len(train),meta_val.shape[1]), dtype=np.float32)
    b_train=np.empty((len(train),3),dtype=np.float32)
    checklist=np.zeros(len(train),dtype=bool)
    for i,(subtrain,subquery) in enumerate(inner):
        r=cached_retrieval(f'o{fold}-i{i}',X,Y,train[subtrain],train[subquery])
        d=deep_cache(f'o{fold}-i{i}',meta,X,Y,train[subtrain],train[subquery],SEED+100+fold*3+i)
        meta_train[subquery]=features(d,r,proj_train[subquery])
        b_train[subquery]=baseline(d,r)
        assert not checklist[subquery].any() and not np.intersect1d(train[subtrain],val).size
        checklist[subquery]=True
    assert checklist.all()
    torch.cuda.empty_cache()
    for name,config in stacks.items():
        print(f'  CatBoost {name} on fold {fold+1}',flush=True)
        outputs[f'stack_{name}']=fit_predict(meta_train,Y[train],meta_val,b_train,baseline(dnn,ret),config,SEED+fold)
    scores={name:metrics(checked_prediction(pred),truth[val]) for name,pred in outputs.items()}
    save_json(summary,scores)
    print(f'Outer fold {fold+1}/5: {perf_counter()-begin:.1f} s',flush=True)
    print({name:round(row['rmse_km'],1) for name,row in scores.items()},flush=True)
    return scores


def collect_cells(fold,X,Y,truth,train,val):
    from cell_v2 import fit_predict
    path=OUT/f'cells-{fold}.json'
    if path.exists():
        return json.loads(path.read_text())
    t=perf_counter()
    p={}
    _,_,meta=config_registry()
    base_dnn=deep_cache(f'o{fold}',meta,X,Y,train,val,SEED+fold)
    ret=cached_retrieval(f'o{fold}',X,Y,train,val)
    base=baseline(base_dnn,ret)
    for k in CELL_COUNTS:
        predictions=fit_predict(X,Y,train,val,k,SEED+200+fold)
        for kind,xyz in predictions.items():
            p[f'cell_{kind}_{k}']=checked_prediction(xyz)
        p[f'cell_hybrid_{k}']=spherical_blend(base,predictions['posterior_mean'],.25)
    scores={name:metrics(pred,truth[val]) for name,pred in p.items()}
    save_json(path,scores)
    print(f'Cell models on fold {fold+1}/5: {perf_counter()-t:.1f} s',flush=True)
    print({name:round(row['rmse_km'],1) for name,row in scores.items()},flush=True)
    return scores


def cv(X,Y,truth,folds):
    scores = [collect_fold(fold,X,Y,truth,train,val,inner) for fold,train,val,inner in make_sections(folds)]
    for fold,train,val,_ in make_sections(folds):
        scores[fold].update(collect_cells(fold,X,Y,truth,train,val))
    ranking=[]
    for name in scores[0]:
        parts=[row[name] for row in scores]
        n=sum(s['n'] for s in parts)
        mse=sum(s['mse_km2']*s['n'] for s in parts)/n
        mae=sum(s['mae_km']*s['n'] for s in parts)/n
        ranking.append({'name':name,'cv_mae_km':mae,'cv_rmse_km':mse**.5,'fold_rmse_km':[s['rmse_km'] for s in parts]})
    ranking.sort(key=lambda row:row['cv_rmse_km'])
    locked={'winner':ranking[0]['name'],'candidates':ranking,'split_sha256':split_fingerprint()}
    save_json(OUT / 'cv-summary.json',locked)
    print(json.dumps(locked,indent=2),flush=True)


def final(X,Y,folds):
    locked=json.loads((OUT/'cv-summary.json').read_text())
    if locked['split_sha256']!=split_fingerprint():
        raise RuntimeError('The split changed after CV.')
    if (OUT/'results.json').exists():
        existing=json.loads((OUT/'results.json').read_text())
        if existing['winner']!=locked['winner']:
            raise RuntimeError('This test was already evaluated for a different winner.')
        print(json.dumps(existing,indent=2),flush=True)
        return
    name=locked['winner']
    print(f'Locked winner: {name}; train rows: {len(X)}',flush=True)
    # Training methods receive only development targets. Holdout targets are
    # opened below, after the winning model's predictions are saved.
    query_features=np.load(DATA/'test_x.npy',mmap_mode='r')
    x=np.concatenate([X,query_features])
    y=np.concatenate([Y,np.full((len(query_features),3),np.nan,dtype=np.float32)])
    train=np.arange(len(X))
    query=np.arange(len(X),len(x))
    deeps,stacks,meta = config_registry()

    if name.startswith('cell_'):
        from cell_v2 import fit_predict
        kind,k=name.rsplit('_',1)
        kind=kind.removeprefix('cell_')
        posterior=fit_predict(x,y,train,query,int(k),SEED+205,save_path=OUT/'best-cells.pt')
        if kind == 'hybrid':
            ret=cached_retrieval('final',x,y,train,query)
            dnn=deep_prediction(meta,x,y,train,query,SEED+5,OUT/'best-base.pt')
            predictions=spherical_blend(baseline(dnn,ret),posterior['posterior_mean'],.25)
        else:
            predictions=posterior[kind]
    elif name.startswith('direct_'):
        predictions=deep_prediction(name.removeprefix('direct_'),x,y,train,query,SEED+5,OUT/'best-base.pt')
    else:
        ret = cached_retrieval('final',x,y,train,query)
        if name in ('retrieval','retrieval_geo_bias'):
            predictions=ret['joint' if name=='retrieval' else 'favor_geo']
        elif name == 'reference_v1':
            predictions=baseline(old_direct(x,y,train,query,417364,OUT/'best-base.pt'),ret)
        elif name == 'hybrid_confidence':
            predictions=geo_adaptive(deep_prediction(meta,x,y,train,query,SEED+5,OUT/'best-base.pt'),ret)
        elif name.startswith('hybrid_'):
            predictions=baseline(deep_prediction(name.removeprefix('hybrid_'),x,y,train,query,SEED+5,OUT/'best-base.pt'),ret)
        elif name.startswith('stack_'):
            from stack_v2 import fit_predict
            direct=deep_prediction(meta,x,y,train,query,SEED+5,OUT/'best-base.pt')
            proj_train,proj_query,center,projection=pca_features(x,train,query,SEED+5)
            atomic_arrays(OUT/'best-pca.npz',center=center,projection=projection)
            f_query=features(direct,ret,proj_query)
            f_train=np.empty((len(train),f_query.shape[1]),dtype=np.float32)
            b_train=np.empty((len(train),3),dtype=np.float32)
            for fold,t,v,_ in make_sections(folds):
                r=cached_retrieval(f'o{fold}',X,Y,t,v)
                d=deep_cache(f'o{fold}',meta,X,Y,t,v,SEED+fold)
                f_train[v]=features(d,r,proj_train[v])
                b_train[v]=baseline(d,r)
            torch.cuda.empty_cache()
            config=stacks[name.removeprefix('stack_')]
            predictions=fit_predict(f_train,Y,f_query,b_train,baseline(direct,ret),config,SEED+5,save_path=OUT/'best-stack.cbm')
        else:
            raise ValueError(name)
    predictions=checked_prediction(predictions)
    atomic_arrays(OUT/'test-predictions.npz',xyz=predictions)
    labels=np.load(DATA/'test_latlon.npy',mmap_mode='r')
    evaluation={'winner':name,'test':metrics(predictions,labels),'cv':locked['candidates'][0],
                'historical_reference':REFERENCE,'split_sha256':locked['split_sha256']}
    save_json(OUT/'results.json',evaluation)
    print(json.dumps(evaluation,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase', choices=('retrieve','cv','final'))
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-capable GPU is required.')
    OUT.mkdir(parents=True,exist_ok=True)
    fingerprint = split_fingerprint()
    marker = OUT / 'split.sha256'
    if marker.exists() and marker.read_text().strip() != fingerprint:
        raise RuntimeError('Cached results come from a different split. Choose an empty run directory.')
    marker.write_text(fingerprint + '\n')
    X,Y,truth,folds=open_dev()
    if args.phase=='retrieve':
        build_retrieval(X,Y,folds)
    elif args.phase=='cv':
        cv(X,Y,truth,folds)
    else:
        locked=OUT/'cv-summary.json'
        if not locked.exists():
            raise RuntimeError('Complete cross-validation before final testing.')
        cv_config=json.loads(locked.read_text())
        if not any(c['name'].startswith('cell_') for c in cv_config['candidates']):
            if (OUT/'results.json').exists():
                raise RuntimeError('This test was already scored. Use a fresh holdout for new candidates.')
            cv(X,Y,truth,folds)
        final(X,Y,folds)


if __name__=='__main__':
    main()
