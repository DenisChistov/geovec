"""Evaluate the literature-inspired first batch on the fixed development folds."""
from time import perf_counter
from pathlib import Path
import json

import numpy as np
import torch

from experiment import metrics, save_json
from run_v3 import DATA, folds_of, fingerprint, unit, blend, pca_features, take_features
from run_v4 import load_bundle, inner_features
from calibration_ideas import predict as calibrate
from literature import METHODS, fit_cells, fit_ranker

OUT=DATA/"v5"
SEED=823748


def per_fold(fold,X,Y,latlon,train,val,inner):
    path=OUT/f"literature-fold-{fold}.json"
    if path.is_file():
        return json.loads(path.read_text())
    scores={}
    preds={}
    start=perf_counter()
    outer=load_bundle(f"o{fold}")
    p_train,p_val,*_=pca_features(X,train,val,723748+fold)
    f_train,e_train=inner_features(outer,p_train,fold,inner)
    f_val=take_features(outer,p_val)
    incumbent=unit(calibrate(f_train,Y[train],f_val,e_train,outer["experts"],
                             723748+fold,only="cal_interaction")["cal_interaction"])
    preds["reference_v4"]=incumbent
    for offset,method in enumerate(METHODS):
        torch.cuda.empty_cache()
        print(f"Fold {fold+1}/5 {method}",flush=True)
        t=perf_counter()
        if method=="candidate_ranker":
            new=fit_ranker({"features":f_train,"experts":e_train},Y[train],
                           {"features":f_val,"experts":outer["experts"]},SEED+fold+offset)
        else:
            new=fit_cells(method,X,Y,train,val,SEED+fold+offset)
        preds[method]=unit(new)
        preds[f"{method}_plus_v4"]=blend(incumbent,new,.33)
        print(f"   completed in {perf_counter()-t:.1f}s",flush=True)
    for key,value in preds.items():
        scores[key]=metrics(value,latlon[val])
    save_json(path,scores)
    np.savez(OUT/f"literature-predictions-{fold}.npz",**preds)
    print(f"Literature fold {fold+1}/5 total {perf_counter()-start:.1f}s",flush=True)
    print({k:round(v["rmse_km"],2) for k,v in scores.items()},flush=True)
    return scores


def main():
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True)
    X=np.load(DATA/"dev_x.npy",mmap_mode="r")
    Y=np.load(DATA/"dev_y.npy",mmap_mode="r")
    latlon=np.load(DATA/"dev_latlon.npy",mmap_mode="r")
    folds=np.load(DATA/"folds.npy")
    assert np.bincount(folds).tolist()==[42500]*5
    assert (DATA/"v3"/"split.sha256").read_text().strip()==fingerprint()
    rows=[per_fold(fold,X,Y,latlon,train,val,inner) for fold,train,val,inner in folds_of(folds)]
    ranking=[]
    for name in rows[0]:
        parts=[r[name] for r in rows]
        n=sum(p["n"] for p in parts)
        ranking.append({"name":name,"mae_km":sum(p["mae_km"]*p["n"] for p in parts)/n,
                        "rmse_km":(sum(p["mse_km2"]*p["n"] for p in parts)/n)**.5,
                        "fold_rmse_km":[p["rmse_km"] for p in parts]})
    ranking.sort(key=lambda x:x["rmse_km"])
    save_json(OUT/"literature-summary.json",{"candidates":ranking,"split_sha256":fingerprint()})
    print(json.dumps(ranking,indent=2),flush=True)


if __name__=="__main__":
    main()
