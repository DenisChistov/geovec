"""Tune the assembled ensemble on all five development folds; never read test labels."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import diversity_cv
import tune_rules
from experiment import metrics,save_json
from full_search import DATA,ROOT
from run_v3 import fingerprint,folds_of

FORMAT=1


def member_ids(selection):
    if isinstance(selection,(tuple,list)):
        return list(selection)
    if 'tuned_fusion' in selection:
        return selection['tuned_fusion']['members']
    if 'diversity_fusion' in selection:
        return selection['diversity_fusion']['members']
    return diversity_cv.member_ids(selection)


def signature(members,split):
    h=hashlib.sha256(json.dumps([FORMAT,members,split,tune_rules.grid()],sort_keys=True).encode())
    for p in (Path(__file__),Path(tune_rules.__file__)):
        h.update(p.read_bytes())
    return h.hexdigest()


def fold_scores(fold,X,Y,ll,train,val,inner,members,split):
    out=ROOT/'tuned'
    out.mkdir(exist_ok=True)
    path=out/f'fold-{fold}.json'
    sig=signature(members,split)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['signature']==sig and set(saved['scores'])==set(tune_rules.grid()):
            return saved['scores']
    t0=perf_counter()
    m,tr,qa=diversity_cv.collect_fold(fold,X,Y,train,val,inner,members)
    assert m==members
    g=tune_rules.geometry(tr,Y[train]);q=tune_rules.geometry(qa)
    rows={}
    for i,(name,value) in enumerate(tune_rules.grouped_predictions(g,q),1):
        rows[name]=metrics(value,ll[val])
        if i%40==0:
            print(f'Tune outer {fold+1}/5: {i}/{len(tune_rules.grid())}, {perf_counter()-t0:.1f}s',flush=True)
    assert set(rows)==set(tune_rules.grid())
    save_json(path,{'signature':sig,'scores':rows})
    print(f'Tune outer {fold+1}/5 completed in {perf_counter()-t0:.1f}s',flush=True)
    print(sorted([(name,round(s['rmse_km'],2)) for name,s in rows.items()],key=lambda x:x[1])[:5],flush=True)
    return rows


def final_prediction(winner,selection,query,Y,folds,save_path=None):
    members=member_ids(selection)
    lock=json.loads((ROOT/'cv-summary.json').read_text())
    assert winner==lock['winner'] and members==lock['tuned_fusion']['members']
    dev=diversity_cv.final_oof(members,folds)
    g=tune_rules.geometry(dev,Y)
    state=tune_rules.fit_one(tune_rules.grid()[winner],g)
    result=tune_rules.predict_geometry(tune_rules.geometry(np.stack([query[k] for k in members],axis=1)),state)
    if save_path is not None:
        path=Path(save_path)
        temporary=path.with_suffix('.tmp.npz')
        params={k:np.asarray(v) for k,v in state.items() if k not in ('kind','mode','context','pool')}
        np.savez(temporary,format=FORMAT,winner=winner,members=np.array(members),**params)
        temporary.replace(path)
        assert np.allclose(result,predict_saved(path,query),rtol=0,atol=2e-6)
    return result


def predict_saved(path,query):
    with np.load(path,allow_pickle=False) as ar:
        if int(ar['format'])!=FORMAT:raise ValueError('Unknown tuning checkpoint format')
        winner=str(ar['winner']);members=ar['members'].tolist()
        state={**tune_rules.grid()[winner]}
        for k in ('center','scale','coef'):
            if k in ar.files:state[k]=ar[k]
    return tune_rules.predict_geometry(tune_rules.geometry(np.stack([query[k] for k in members],axis=1)),state)


def main():
    lock_path=ROOT/'cv-summary.json'
    original=json.loads(lock_path.read_text())
    assert 'diversity_fusion' in original,'Complete diversity_cv.py first'
    split=fingerprint()
    assert original['split_sha256']==split
    members=member_ids(original)
    specs=tune_rules.grid()
    done=original.get('tuned_fusion')
    if done and done['rules']==specs and done['members']==members:
        print('Detailed tuning already completed:',original['winner'],flush=True);return
    X,Y,ll,folds=[np.load(DATA/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in ('dev_x','dev_y','dev_latlon','folds')]
    assert len(X)==len(Y)==len(ll)==len(folds)==212500
    parts=[fold_scores(f,X,Y,ll,t,v,i,members,split) for f,t,v,i in folds_of(folds)]
    new=[]
    for name in specs:
        p=[row[name] for row in parts];n=sum(r['n'] for r in p)
        new.append({'name':name,'cv_mae_km':sum(r['mae_km']*r['n'] for r in p)/n,
                    'cv_rmse_km':(sum(r['mse_km2']*r['n'] for r in p)/n)**.5,
                    'fold_rmse_km':[r['rmse_km'] for r in p]})
    old=[r for r in original['candidates'] if not r['name'].startswith('tuned:')]
    candidates=sorted(old+new,key=lambda r:(r['cv_rmse_km'],r['name']))
    if any((ROOT/x).exists() for x in ('final-provenance.json','test-predictions.npz','results.json')):
        raise RuntimeError('Final test was already committed; cannot tune further on this lock')
    if json.loads(lock_path.read_text())!=original:raise RuntimeError('CV changed during tuning')
    merged={**original,'candidates':candidates,'winner':candidates[0]['name'],
            'tuned_fusion':{'format':FORMAT,'members':members,'rules':specs,'n_variants':len(specs),
                            'previous_cv_winner':old[0]['name'],'selection_metric':'pooled WGS84 RMSE',
                            'stage':'five-fold development tuning; unchanged historical test'}}
    save_json(lock_path,merged)
    save_json(ROOT/'tuned'/'summary.json',{'candidates':sorted(new,key=lambda r:r['cv_rmse_km']),'winners':candidates[:10],'specifications':specs})
    print(f'TUNING COMPLETE: {len(specs)} × 5; winner {merged["winner"]}',flush=True)
    print(json.dumps(candidates[:10],indent=2),flush=True)


if __name__=='__main__':main()
