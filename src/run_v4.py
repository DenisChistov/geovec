"""Fifty diverse second-stage experiments with five disjoint outer folds."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiment import metrics, save_json
from run_v3 import folds_of, fingerprint, take_features, unit, pca_features

DATA = Path(os.environ.get("GEO_CV_ROOT", "/content/train250k-cv"))
V3 = DATA / "v3"
OUT = DATA / "v4"
SEED = 723748


def modules():
    import posterior_ideas, density_ideas, learned_ideas, calibration_ideas
    return (posterior_ideas,density_ideas),(learned_ideas,calibration_ideas)


def registry():
    direct,learned = modules()
    by_name = {}
    for group,kind in ((direct,"deterministic"),(learned,"trained")):
        for module in group:
            for name, description in module.DESCRIPTIONS.items():
                if name in by_name:
                    raise ValueError(f"Duplicate method name: {name}")
                by_name[name]=(module,kind,description)
    assert len(by_name)==50, len(by_name)
    return by_name


def load_bundle(key):
    with np.load(V3 / f"bundle-{key}.npz",allow_pickle=False) as source:
        return dict(source)


def inner_features(outer, p_train, fold, inner):
    count=len(p_train)
    x=np.empty((count,outer["features"].shape[1]+p_train.shape[1]),dtype=np.float32)
    e=np.empty((count,*outer["experts"].shape[1:]),dtype=np.float32)
    seen=np.zeros(count,dtype=bool)
    for i, (_,q) in enumerate(inner):
        b=load_bundle(f"o{fold}-i{i}")
        assert not seen[q].any()
        x[q]=take_features(b,p_train[q])
        e[q]=b["experts"]
        seen[q]=True
    assert seen.all()
    return x,e


def record_scores(path, predictions, truth, expected_names):
    assert set(predictions)==set(expected_names)
    checked={}
    for name, value in predictions.items():
        checked[name]=metrics(unit(value),truth)
    save_json(path,checked)
    return checked


def read_or_compute(path, calculate, truth, names):
    if path.exists():
        rows=json.loads(path.read_text())
        assert set(rows)==set(names)
        return rows
    return record_scores(path, calculate(),truth,names)


def run_fold(fold,X,Y,latlon,train,val,inner):
    t0=perf_counter()
    direct,learned=modules()
    outer=load_bundle(f"o{fold}")
    truth=latlon[val]
    scores={}
    for module in direct:
        start=perf_counter()
        scores.update(read_or_compute(
            OUT/f"{module.__name__}-{fold}.json", lambda:module.predict(outer),
            truth,module.DESCRIPTIONS))
        print(f"{module.__name__} fold {fold+1}: {perf_counter()-start:.1f}s",flush=True)
    missing=[module for module in learned if not (OUT/f"{module.__name__}-{fold}.json").exists()]
    if missing:
        p_train,p_val,*_=pca_features(X,train,val,SEED+fold)
        train_f,train_e=inner_features(outer,p_train,fold,inner)
        val_f=take_features(outer,p_val)
        torch.cuda.empty_cache()
    for module in learned:
        start=perf_counter()
        path=OUT/f"{module.__name__}-{fold}.json"
        if path.exists():
            local=json.loads(path.read_text())
        else:
            local=record_scores(path,module.predict(
                train_f,Y[train],val_f,train_e,outer["experts"],SEED+fold),
                truth,module.DESCRIPTIONS)
        assert set(local)==set(module.DESCRIPTIONS)
        scores.update(local)
        print(f"{module.__name__} fold {fold+1}: {perf_counter()-start:.1f}s",flush=True)
    assert len(scores)==50
    save_json(OUT/f"fold-{fold}.json",scores)
    print(f"Completed 50 ideas on fold {fold+1}/5: {perf_counter()-t0:.1f}s",flush=True)
    print([(k,round(v["rmse_km"],2)) for k,v in sorted(scores.items(),key=lambda kv:kv[1]["rmse_km"])[:5]],flush=True)
    return scores


def cross_validate(X,Y,latlon,folds):
    by_name=registry()
    rows=[run_fold(fold,X,Y,latlon,train,val,inner)
          for fold,train,val,inner in folds_of(folds)]
    leaderboard=[]
    for name,(module,kind,desc) in by_name.items():
        parts=[r[name] for r in rows]
        n=sum(p["n"] for p in parts)
        mae=sum(p["mae_km"]*p["n"] for p in parts)/n
        mse=sum(p["mse_km2"]*p["n"] for p in parts)/n
        leaderboard.append({"name":name,"description":desc,"family":module.__name__,
                            "kind":kind,"cv_mae_km":mae,"cv_rmse_km":mse**.5,
                            "fold_rmse_km":[p["rmse_km"] for p in parts]})
    previous=json.loads((V3/"cv-summary.json").read_text())
    assert previous["split_sha256"]==fingerprint()
    incumbent={**previous["candidates"][0],"original_name":previous["winner"],
               "name":"incumbent_v3","description":"Previously selected V3 ensemble",
               "family":"incumbent","kind":"incumbent"}
    leaderboard.append(incumbent)
    leaderboard.sort(key=lambda row:row["cv_rmse_km"])
    result={"winner":leaderboard[0]["name"],"n_new_variants":50,"candidates":leaderboard,
            "split_sha256":fingerprint()}
    save_json(OUT/"cv-summary.json",result)
    print("New winner selection:",result["winner"],flush=True)
    print(json.dumps(leaderboard[:5],indent=2),flush=True)


def final(X,Y,folds):
    locked=json.loads((OUT/"cv-summary.json").read_text())
    assert locked["split_sha256"]==fingerprint()
    winner=locked["winner"]
    if (OUT/"results.json").exists():
        result=json.loads((OUT/"results.json").read_text())
        assert result["winner"]==winner
        print(json.dumps(result,indent=2))
        return
    historical=json.loads((V3/"results.json").read_text())
    if winner=="incumbent_v3":
        with np.load(V3/"test-predictions.npz") as saved:
            prediction=saved["xyz"]
        test=historical["test"]
    else:
        module,kind,_=registry()[winner]
        bundle=load_bundle("final")
        if kind=="deterministic":
            prediction=unit(module.predict(bundle)[winner])
        else:
            test_x=np.load(DATA/"test_x.npy",mmap_mode="r")
            xx=np.concatenate([X,test_x])
            train=np.arange(len(X))
            test_ids=np.arange(len(X),len(xx))
            p_train,p_test,center,projection=pca_features(xx,train,test_ids,SEED+5)
            np.savez(OUT/"best-pca.npz",center=center,projection=projection)
            f_test=take_features(bundle,p_test)
            f_train=np.empty((len(train),f_test.shape[1]),np.float32)
            e_train=np.empty((len(train),*bundle["experts"].shape[1:]),np.float32)
            for fold,_,val,_ in folds_of(folds):
                b=load_bundle(f"o{fold}")
                f_train[val]=take_features(b,p_train[val])
                e_train[val]=b["experts"]
            torch.cuda.empty_cache()
            prediction=unit(module.predict(f_train,Y,f_test,e_train,bundle["experts"],
                                          SEED+5,save_dir=OUT,only=winner)[winner])
        # Scores are computed only after model choice and final stage training.
        test=metrics(prediction,np.load(DATA/"test_latlon.npy",mmap_mode="r"))
    np.savez(OUT/"test-predictions.npz",xyz=prediction)
    result={"winner":winner,"test":test,"cv":locked["candidates"][0],
            "historical_reference":historical["test"],"split_sha256":fingerprint()}
    save_json(OUT/"results.json",result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    cmd=argparse.ArgumentParser()
    cmd.add_argument("phase",choices=("direct","cv","final"))
    args=cmd.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available(), "Colab GPU required"
    assert (V3/"split.sha256").read_text().strip()==fingerprint()
    X=np.load(DATA/"dev_x.npy",mmap_mode="r")
    Y=np.load(DATA/"dev_y.npy",mmap_mode="r")
    folds=np.load(DATA/"folds.npy")
    assert np.bincount(folds).tolist()==[42500]*5
    if args.phase=="direct":
        import posterior_ideas,density_ideas
        latlon=np.load(DATA/"dev_latlon.npy",mmap_mode="r")
        for fold,_,val,_ in folds_of(folds):
            bundle=load_bundle(f"o{fold}")
            for module in (posterior_ideas,density_ideas):
                read_or_compute(OUT/f"{module.__name__}-{fold}.json",
                    lambda:module.predict(bundle),latlon[val],module.DESCRIPTIONS)
            print(f"28 deterministic ideas evaluated on fold {fold+1}/5",flush=True)
    elif args.phase=="cv":
        cross_validate(X,Y,np.load(DATA/"dev_latlon.npy",mmap_mode="r"),folds)
    else:
        final(X,Y,folds)


if __name__=="__main__":
    main()
