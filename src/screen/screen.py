"""Staged 500-configuration screening; no final holdout is opened."""
import sys
from pathlib import Path
import json
import os
from time import perf_counter

import numpy as np
import torch

DATA=Path(os.environ.get("GEO_CV_ROOT","/content/train250k-cv"))
sys.path.insert(0,str(DATA))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from experiment import metrics,save_json
from run_v3 import unit,folds_of,pca_features,take_features
from run_v4 import load_bundle,inner_features
from calibration_ideas import predict as v4_predict

import features,neural_models,geo_models,retrieval_models,meta_models,ensemble_models

OUT=DATA/"screen500"
SEED=904317


def ident(family,view,method):
    return f"{family}/{view}/{method}"


def configs():
    out={}
    for family,models in (("neural",neural_models),("geo",geo_models),("retrieval",retrieval_models)):
        for view,feature_description in features.FEATURES.items():
            for method,description in models.METHODS.items():
                out[ident(family,view,method)]={"family":family,"features":view,"method":method,
                    "feature_description":feature_description,"model_description":description}
    for family,views,models in (("meta",meta_models.VIEWS,meta_models.METHODS),("ensemble",ensemble_models.GROUPS,ensemble_models.METHODS)):
        for view,description in views.items():
            for method,model_description in models.items():
                out[ident(family,view,method)]={"family":family,"features":view,"method":method,
                    "feature_description":description,"model_description":model_description}
    assert len(out)==500, len(out)
    return out


def path_for(name,ext):
    return OUT/(name.replace("/","__")+ext)


def done(name,need_predictions=False):
    return path_for(name,".json").is_file() and (not need_predictions or path_for(name,".npy").is_file())


def record(name,pred,truth,n_cal,seconds,keep=False):
    pred=unit(pred)
    s=metrics(pred[n_cal:],truth)
    if keep:
        p=path_for(name,".tmp.npy")
        np.save(p,pred)
        p.replace(path_for(name,".npy"))
    row={"name":name,"seconds":round(seconds,2),**s}
    save_json(path_for(name,".json"),row)
    print(f"{name}: RMSE={s['rmse_km']:.2f}, MAE={s['mae_km']:.2f}, {seconds:.2f}s",flush=True)


def fail(name,error):
    save_json(path_for(name,".error.json"),{"name":name,"error":type(error).__name__+": "+str(error)})
    print(f"FAILED {name}: {type(error).__name__}: {error}",flush=True)
    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    OUT.mkdir(exist_ok=True)
    catalog=configs()
    save_json(OUT/"registry.json",catalog)
    X=np.load(DATA/"dev_x.npy",mmap_mode="r")
    Y=np.load(DATA/"dev_y.npy",mmap_mode="r")
    latlon=np.load(DATA/"dev_latlon.npy",mmap_mode="r")
    folds=np.load(DATA/"folds.npy")
    fold,train,val,inner=next(iter(folds_of(folds)))
    rng=np.random.default_rng(SEED)
    rng_train=rng.permutation(train)
    fit=np.sort(rng_train[:60000])
    cal=np.sort(rng_train[60000:80000])
    evaluate=np.sort(rng.choice(val,12500,replace=False))
    q=np.concatenate([cal,evaluate])
    assert not np.intersect1d(fit,cal).size and not np.intersect1d(q,fit).size
    np.savez(OUT/"split.npz",fit=fit,cal=cal,evaluate=evaluate)
    n_cal=len(cal)
    truth=latlon[evaluate]
    raw_train,raw_query=X[fit],X[q]
    prepared=geo_models.prepare(Y[fit],SEED)
    for view,(a,b) in features.prepare(raw_train,Y[fit],raw_query,SEED):
        print(f"FEATURE {view}: {a.shape[1]} dimensions",flush=True)
        names=[ident("retrieval",view,m) for m in retrieval_models.METHODS]
        if not all(done(name,True) for name in names):
            try:
                t=perf_counter()
                preds=retrieval_models.predict(a,Y[fit],b,SEED)
                assert set(preds)==set(retrieval_models.METHODS)
                elapsed=(perf_counter()-t)/len(preds)
                for method,value in preds.items():
                    name=ident("retrieval",view,method)
                    record(name,value,truth,n_cal,elapsed,keep=True)
            except Exception as exc:
                for name in names:
                    if not done(name,True):
                        fail(name,exc)
        for family,module in (("neural",neural_models),("geo",geo_models)):
            for method in module.METHODS:
                name=ident(family,view,method)
                if done(name,True):
                    continue
                try:
                    t=perf_counter()
                    if family=="neural":
                        value=module.fit_predict(method,a,Y[fit],b,SEED,epochs=6)
                    else:
                        value=module.predict(method,a,Y[fit],b,prepared,SEED)
                    record(name,value,truth,n_cal,perf_counter()-t,keep=True)
                except Exception as exc:
                    fail(name,exc)
                    raise
    base_names=[name for name,spec in catalog.items() if spec["family"] in ("neural","geo","retrieval")]
    assert all(done(name,True) for name in base_names), "Recover missing base predictions"
    outer=load_bundle("o0")
    p_train,p_val,*_=pca_features(X,train,val,SEED)
    f_train,e_train=inner_features(outer,p_train,fold,inner)
    f_val=take_features(outer,p_val)
    pos_fit=np.searchsorted(train,fit)
    pos_cal=np.searchsorted(train,cal)
    pos_val=np.searchsorted(val,evaluate)
    c_train=f_train[pos_cal]
    c_val=f_val[pos_val]
    e_cal=e_train[pos_cal]
    e_val=outer["experts"][pos_val]
    v4_query=np.concatenate([c_train,c_val],axis=0)
    v4_experts=np.concatenate([e_cal,e_val],axis=0)
    v4_prior=v4_predict(f_train[pos_fit],Y[fit],v4_query,e_train[pos_fit],v4_experts,SEED,
                        only="cal_interaction")["cal_interaction"]
    prior_ref=v4_predict(f_train,Y[train],c_val,e_train,e_val,723748,only="cal_interaction")["cal_interaction"]
    save_json(OUT/"reference-v4.json",metrics(prior_ref,truth))
    for (view,a),(v,b) in zip(meta_models.views(c_train,e_cal),meta_models.views(c_val,e_val)):
        assert view==v
        for method in meta_models.METHODS:
            name=ident("meta",view,method)
            if done(name):
                continue
            t=perf_counter()
            try:
                p=meta_models.fit_predict(method,a,Y[cal],b,e_cal,e_val,SEED)
                record(name,p,truth,0,perf_counter()-t)
            except Exception as exc:
                fail(name,exc)
                raise
    predictions=[np.load(path_for(name,".npy")) for name in base_names]
    predictions.append(unit(v4_prior))
    base_names.append("prior/prior/v4")
    all_preds=np.stack(predictions,axis=1)
    del predictions
    c_preds,eval_preds=all_preds[:n_cal],all_preds[n_cal:]
    groups=ensemble_models.select_groups(base_names,c_preds,Y[cal])
    selected={name:[base_names[i] for i in ids] for name,ids in groups.items()}
    save_json(OUT/"ensemble-components.json",selected)
    for group,ids in groups.items():
        for method in ensemble_models.METHODS:
            name=ident("ensemble",group,method)
            if done(name):
                continue
            t=perf_counter()
            try:
                value=ensemble_models.one(method,c_preds[:,ids],Y[cal],eval_preds[:,ids])
                record(name,value,truth,0,perf_counter()-t)
            except Exception as exc:
                fail(name,exc)
                raise
    rows=[json.loads(path_for(name,".json").read_text()) for name in catalog]
    assert len(rows)==500
    rows.sort(key=lambda r:r["rmse_km"])
    save_json(OUT/"summary.json",{"candidates":rows,"n":500,"reference_v4":json.loads((OUT/"reference-v4.json").read_text())})
    print("SCREEN_COMPLETE",flush=True)
    print(json.dumps(rows[:10],indent=2),flush=True)


if __name__=="__main__":
    main()
