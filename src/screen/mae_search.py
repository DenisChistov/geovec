"""Separate WGS84 MAE model selection. The existing RMSE lock is read-only."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

import full_search
import diversity_cv
import tune_rules
import mae_rules
import mae_posterior
import mae_gate
import mae_poststage
from experiment import metrics,save_json
from run_v3 import fingerprint,folds_of,unit,blend
from run_v4 import load_bundle

DATA=full_search.DATA
RMSE=full_search.ROOT
OUT=DATA/'mae_cv'
SEED=1562391


def rmse_lock():
    path=RMSE/'cv-summary.json'
    b=path.read_bytes();lock=json.loads(b)
    if lock['split_sha256']!=fingerprint() or 'tuned_fusion' not in lock:
        raise ValueError('Missing or incompatible RMSE CV lock')
    return lock,hashlib.sha256(b).hexdigest()


def registry(members):
    specs={}
    for name,rule in mae_posterior.grid().items():
        specs['posterior:'+name]={'kind':'posterior','method':name,'parameters':rule}
    for name,rule in mae_rules.grid().items():
        specs['robust-base:'+name]={'kind':'robust','pool':'base','method':name,'parameters':rule}
        specs['robust-median:'+name]={'kind':'robust','pool':'median','method':name,'parameters':rule}
    for name,rule in mae_gate.grid().items():
        specs['gate:'+name]={'kind':'gate','method':name,'parameters':rule}
    for p in mae_posterior.GATE_EXPERTS:
        for j,k in enumerate(members):
            for weight in (.05,.15,.30):
                specs[f'blend:{p}/{j}/w{weight:g}']={'kind':'blend','posterior':p,'expert':k,'weight':weight}
    return specs


def signature(members,split,rmse_sha):
    h=hashlib.sha256(json.dumps([members,split,rmse_sha,registry(members)],sort_keys=True).encode())
    for mod in (mae_rules,mae_rules.mae_median,mae_posterior,mae_gate,mae_poststage):h.update(Path(mod.__file__).read_bytes())
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()


def post_gate(bundle):
    return mae_posterior.predictions(bundle,only=mae_posterior.GATE_EXPERTS)


def directions(base,bundle,post):
    # V3's seven experts already appear in diagnostic features. Keep them in
    # the explicit candidate pool once; the diagnostic matrix is context only.
    return np.concatenate([base,bundle['experts'],np.stack([post[k] for k in mae_posterior.GATE_EXPERTS],axis=1)],axis=1).astype(np.float32)


def median_pool(base,post):
    return np.concatenate([post[mae_posterior.GATE_EXPERTS[0]][:,None,:],base],axis=1)


def build_inner(fold,train,inner,base):
    m=len(train)
    out=None;f=None;med=None;seen=np.zeros(m,bool)
    for j,(_,q) in enumerate(inner):
        if seen[q].any():raise ValueError('Duplicate inner rows')
        b=load_bundle(f'o{fold}-i{j}')
        if len(b['features'])!=len(q):raise ValueError('Inner bundle row count mismatch')
        post,_=mae_poststage.cache(f'o{fold}-i{j}',b);table=directions(base[q],b,post)
        if out is None:
            out=np.empty((m,table.shape[1],3),np.float32)
            f=np.empty((m,b['features'].shape[1]),np.float32)
            med=np.empty((m,base.shape[1]+1,3),np.float32)
        out[q]=table;f[q]=b['features'];med[q]=median_pool(base[q],post);seen[q]=True
    if not seen.all():raise ValueError('Incomplete inner OOF bundle data')
    return out,f,med


def build_final_dev(base,folds):
    m=len(folds);out=None;f=None;med=None
    for fold,_,v,_ in folds_of(folds):
        b=load_bundle(f'o{fold}');post,_=mae_poststage.cache(f'o{fold}',b);table=directions(base[v],b,post)
        if out is None:
            out=np.empty((m,table.shape[1],3),np.float32)
            f=np.empty((m,b['features'].shape[1]),np.float32)
            med=np.empty((m,base.shape[1]+1,3),np.float32)
        out[v]=table;f[v]=b['features'];med[v]=median_pool(base[v],post)
    return out,f,med


def gate_features(f):
    return {'v3_features':f} # add leading posterior centres, not duplicate V3 experts


def _record(rows,name,pred,truth):
    if name in rows:raise ValueError('Duplicate candidate '+name)
    rows[name]=metrics(unit(pred),truth)


def outer_fold(fold,X,Y,ll,train,val,inner,members,split,rmse_sha):
    sig=signature(members,split,rmse_sha)
    path=OUT/f'fold-{fold}.json'
    if path.exists():
        p=json.loads(path.read_text())
        if p['signature']==sig and set(p['scores'])==set(registry(members)):return p['scores']
    t0=perf_counter();rows={};truth=ll[val]
    m,tr,qa=diversity_cv.collect_fold(fold,X,Y,train,val,inner,members)
    assert m==members
    b=load_bundle(f'o{fold}')
    pp,pscores=mae_poststage.cache(f'o{fold}',b,truth)
    assert set(pscores)==set(mae_posterior.grid())
    for name,value in pscores.items():rows['posterior:'+name]=value
    for p in mae_posterior.GATE_EXPERTS:
        for j,k in enumerate(members):
            for weight in (.05,.15,.30):
                _record(rows,f'blend:{p}/{j}/w{weight:g}',blend(pp[p],qa[:,j],weight),truth)
    print(f'MAE fold {fold+1}/5 posterior and blends {len(rows)} variants, {perf_counter()-t0:.1f}s',flush=True)
    train_gate,train_f,train_med=build_inner(fold,train,inner,tr)
    query_gate=directions(qa,b,pp);query_med=median_pool(qa,pp)
    for title,fit,query in [('base',tr,qa),('median',train_med,query_med)]:
        g=tune_rules.geometry(fit,Y[train]);q=tune_rules.geometry(query)
        for i,(name,value) in enumerate(mae_rules.grouped_predictions(g,q),1):
            _record(rows,'robust-'+title+':'+name,value,truth)
            if i%30==0:print(f'MAE fold {fold+1}/5 robust-{title} {i}/{len(mae_rules.grid())}, {perf_counter()-t0:.1f}s',flush=True)
        del g,q
        torch.cuda.empty_cache()
    states=mae_gate.fit_grid(train_gate,Y[train],gate_features(train_f),seed=SEED+fold)
    for name,value in mae_gate.predict_grid(states,query_gate,gate_features(b['features'])).items():
        _record(rows,'gate:'+name,value,truth)
    assert set(rows)==set(registry(members)),(set(registry(members))-set(rows),set(rows)-set(registry(members)))
    save_json(path,{'signature':sig,'scores':rows})
    print(f'MAE fold {fold+1}/5 finished, {len(rows)} variants, {perf_counter()-t0:.1f}s',flush=True)
    print(sorted([(k,round(v['mae_km'],2),round(v['rmse_km'],2)) for k,v in rows.items()],key=lambda x:x[1])[:6],flush=True)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    lock,rmse_sha=rmse_lock();members=lock['tuned_fusion']['members'];split=fingerprint()
    specs=registry(members);sig=signature(members,split,rmse_sha)
    if (OUT/'cv-summary.json').exists():
        prior=json.loads((OUT/'cv-summary.json').read_text())
        if prior['signature']==sig:print('MAE CV already completed',prior['winner'],flush=True);return
        raise RuntimeError('An incompatible MAE selection already exists; keep its historical artifacts')
    if any((OUT/x).exists() for x in ('test-predictions.npz','results.json','final-provenance.json')):
        raise RuntimeError('MAE final evaluation already committed')
    X,Y,ll,folds=[np.load(DATA/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in ('dev_x','dev_y','dev_latlon','folds')]
    assert len(X)==len(Y)==len(ll)==len(folds)==212500
    parts=[outer_fold(f,X,Y,ll,t,v,i,members,split,rmse_sha) for f,t,v,i in folds_of(folds)]
    ranking=[]
    for name in specs:
        p=[r[name] for r in parts];n=sum(x['n'] for x in p)
        ranking.append({'name':name,'cv_mae_km':sum(x['mae_km']*x['n'] for x in p)/n,
                        'cv_rmse_km':(sum(x['mse_km2']*x['n'] for x in p)/n)**.5,
                        'fold_mae_km':[x['mae_km'] for x in p],'fold_rmse_km':[x['rmse_km'] for x in p]})
    ranking.sort(key=lambda r:(r['cv_mae_km'],r['cv_rmse_km'],r['name']))
    if rmse_lock()[1]!=rmse_sha:raise RuntimeError('RMSE lock changed during MAE search')
    reference_v4=json.loads((DATA/'v4'/'cv-summary.json').read_text())
    v4_min=min(reference_v4['candidates'],key=lambda r:r['cv_mae_km'])
    if ranking[0]['cv_mae_km']>v4_min['cv_mae_km']+.02:
        raise RuntimeError('The previously measured strongest MAE posterior must be included and reproduced')
    result={'winner':ranking[0]['name'],'candidates':ranking,'specifications':specs,'members':members,
            'posterior_gate_experts':list(mae_posterior.GATE_EXPERTS),'n_variants':len(ranking),
            'split_sha256':split,'rmse_cv_sha256':rmse_sha,'signature':sig,
            'selection_metric':'pooled exact WGS84 MAE, then RMSE tie-break',
            'references':{'rmse_winner':lock['candidates'][0],'old_v4_mae_best':v4_min,
                          'old_full_mae_best':min(lock['candidates'],key=lambda r:r['cv_mae_km'])}}
    save_json(OUT/'cv-summary.json',result)
    print(f'MAE CV COMPLETE {len(ranking)} × 5, winner {result["winner"]}',flush=True)
    print(json.dumps(ranking[:10],indent=2),flush=True)


if __name__=='__main__':main()
