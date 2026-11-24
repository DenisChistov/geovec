"""Full five-fold selection of screened finalists and cross-fitted final blends."""
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from screen import DATA, OUT, ident
from experiment import metrics,save_json
from run_v3 import folds_of,pca_features,take_features,unit,blend,fingerprint
from run_v4 import load_bundle,inner_features
from calibration_ideas import predict as calibrate
from literature import fit_cells,fit_ranker
import features,neural_models,geo_models,retrieval_models,meta_models,ensemble_models

ROOT=DATA/"full500"
SEED=954317
PRIOR="prior/prior/v4"
PAPER="paper/distance_risk"
JOINT_RULES=("signed_wgs84","simplex","context_gate","regional_risk","inverse_risk")


def choose():
    s=json.loads((OUT/"summary.json").read_text())
    assert s["n"]==500
    order=[row["name"] for row in s["candidates"]]
    champions={f:next(n for n in order if n.startswith(f+"/")) for f in
               ("neural","geo","retrieval","meta","ensemble")}
    finalists=list(dict.fromkeys([*champions.values(),*order[:5]]))
    groups=json.loads((OUT/"ensemble-components.json").read_text())
    own={n:groups[n.split("/")[1]] for n in finalists if n.startswith("ensemble/")}
    joint=[PRIOR,*[champions[k] for k in ("neural","geo","retrieval","meta")],PAPER]
    required=list(dict.fromkeys([PRIOR,*[x for x in finalists if x not in own],*joint,
                                 *[x for group in own.values() for x in group]]))
    return {"finalists":finalists,"champions":champions,"screen_ensembles":own,
            "joint_members":joint,"required":required}


def predict_split(keys,X,Y,train,query,f_train,e_train,f_query,e_query,seed,save_dir=None):
    result={}
    base=[key for key in keys if key.split("/")[0] in ("neural","geo","retrieval")]
    wanted_views=list(dict.fromkeys(key.split("/")[1] for key in base))
    if base:
        x,q=X[train],X[query]
        geo=geo_models.prepare(Y[train],seed) if any(k.startswith("geo/") for k in base) else None
        for view,(a,b) in features.prepare(x,Y[train],q,seed,only=wanted_views,save_dir=save_dir):
            selected=[k for k in base if k.split("/")[1]==view]
            if any(k.startswith("retrieval/") for k in selected):
                pred=retrieval_models.predict(a,Y[train],b,seed,
                    save_path=save_dir/f"retrieval-{view}.pt" if save_dir else None)
                for k in selected:
                    family,_,method=k.split("/")
                    if family=="retrieval":
                        result[k]=unit(pred[method])
            for k in selected:
                family,_,method=k.split("/")
                destination=save_dir/k.replace("/","__") if save_dir else None
                if family=="neural":
                    result[k]=unit(neural_models.fit_predict(method,a,Y[train],b,seed,epochs=6,
                                         save_path=str(destination)+".pt" if destination else None))
                elif family=="geo":
                    if destination:
                        destination.mkdir(parents=True,exist_ok=True)
                    result[k]=unit(geo_models.predict(method,a,Y[train],b,geo,seed,
                                        save_dir=destination))
    if PRIOR in keys:
        result[PRIOR]=unit(calibrate(f_train,Y[train],f_query,e_train,e_query,723748,
                                  only="cal_interaction",save_dir=save_dir)["cal_interaction"])
    meta=[key for key in keys if key.startswith("meta/")]
    if meta:
        for (view,a),(view_q,b) in zip(meta_models.views(f_train,e_train),meta_models.views(f_query,e_query)):
            assert view==view_q
            for key in [key for key in meta if key.split("/")[1]==view]:
                _,_,method=key.split("/")
                dest=save_dir/(key.replace("/","__")+".pt") if save_dir else None
                result[key]=unit(meta_models.fit_predict(method,a,Y[train],b,e_train,e_query,seed,save_path=dest))
    if PAPER in keys:
        torch.cuda.empty_cache()
        result[PAPER]=unit(fit_cells("distance_risk",X,Y,train,query,seed,
                                    save_path=save_dir/"paper-distance-risk.pt" if save_dir else None))
    assert set(result)==set(keys),(set(keys)-set(result))
    return result


def stack_rule(method,member_ids,train_predictions,query_predictions,Y):
    return unit(ensemble_models.one(method,np.stack([train_predictions[k] for k in member_ids],axis=1),
                                    Y,np.stack([query_predictions[k] for k in member_ids],axis=1)))


def checked_cache(path,keys,rows):
    with np.load(path,allow_pickle=False) as saved:
        result=dict(saved)
    if set(result)!=set(keys):
        raise ValueError(f"Prediction cache {path.name} has a different specialist set")
    for name,prediction in result.items():
        if prediction.shape!=(rows,3) or not np.isfinite(prediction).all():
            raise ValueError(f"Bad prediction shape or nonfinite values in {path.name}: {name}")
    return result


def write_predictions(path,predictions):
    temporary=path.with_suffix('.tmp.npz')
    np.savez(temporary,**predictions)
    temporary.replace(path)


def trainable_members(ids):
    # V4 and meta experts already consume cross-fitted base predictions. A
    # second learned gate would need an additional nested cross-fit of V3.
    # Use predetermined weights for those experts, never in learned gates.
    return [k for k in ids if k!=PRIOR and not k.startswith("meta/")]


def apply_fusion(method,ids,train_predictions,query_predictions,Y,joint=False):
    raw=trainable_members(ids)
    assert raw
    fixed=[k for k in ids if k not in raw]
    prediction=stack_rule(method,raw,train_predictions,query_predictions,Y)
    if joint:
        # Final blend is 1/2 raw gate, 1/6 meta, 1/3 incumbent.
        assert PRIOR in fixed and len(fixed)==2
        prediction=blend(prediction,query_predictions[next(k for k in fixed if k!=PRIOR)],.25)
        return blend(prediction,query_predictions[PRIOR],1/3)
    for j,key in enumerate(fixed):
        prediction=blend(prediction,query_predictions[key],1/(len(raw)+j+1))
    return prediction


def outer_fold(selection,fold,X,Y,latlon,train,val,inner):
    path=ROOT/f"fold-{fold}.json"
    if path.exists():
        return json.loads(path.read_text())
    t=perf_counter()
    outer=load_bundle(f"o{fold}")
    p_train,p_val,*_=pca_features(X,train,val,SEED+fold)
    ft,et=inner_features(outer,p_train,fold,inner)
    fv=take_features(outer,p_val);ev=outer["experts"]
    needed=selection["required"]
    outer_path=ROOT/f"base-outer-{fold}.npz"
    if outer_path.exists():
        query=checked_cache(outer_path,needed,len(val))
    else:
        query=predict_split(needed,X,Y,train,val,ft,et,fv,ev,SEED+fold)
        write_predictions(outer_path,query)
    members=trainable_members(list(dict.fromkeys([*selection["joint_members"],*[x for g in selection["screen_ensembles"].values() for x in g]])))
    nested={}
    for key in members:
        nested[key]=np.empty((len(train),3),dtype=np.float32)
    for i,(a,b) in enumerate(inner):
        subpath=ROOT/f"base-inner-{fold}-{i}.npz"
        if subpath.exists():
            local=checked_cache(subpath,members,len(b))
        else:
            print(f"Full fold {fold+1}: OOF specialists inner {i+1}/3",flush=True)
            local=predict_split(members,X,Y,train[a],train[b],None,None,None,None,
                                SEED+100+fold*3+i)
            write_predictions(subpath,local)
        for key,value in local.items():
            nested[key][b]=value
    predictions={"reference_v4":query[PRIOR]}
    for name in selection["finalists"]:
        if name in selection["screen_ensembles"]:
            predicted=apply_fusion(name.rsplit("/",1)[1],selection["screen_ensembles"][name],nested,query,Y[train])
        else:
            predicted=query[name]
        predictions[name]=predicted
        predictions["blend:"+name]=blend(query[PRIOR],predicted,.33)
    for rule in JOINT_RULES:
        predictions["joint:"+rule]=apply_fusion(rule,selection["joint_members"],nested,query,Y[train],joint=True)
    for name in ("distance_smoothing","distance_risk","hierarchical_cells","fourier_location","candidate_ranker"):
        with np.load(DATA/"v5"/f"literature-predictions-{fold}.npz") as saved:
            predictions["literature:"+name]=saved[name]
            predictions["literature:"+name+"_plus_v4"]=saved[name+"_plus_v4"]
    scores={name:metrics(unit(p),latlon[val]) for name,p in predictions.items()}
    save_json(path,scores)
    print(f"Full fold {fold+1}/5 complete {perf_counter()-t:.1f}s",flush=True)
    print([(k,round(v["rmse_km"],2)) for k,v in sorted(scores.items(),key=lambda kv:kv[1]["rmse_km"])[:8]],flush=True)
    return scores


def main():
    ROOT.mkdir(exist_ok=True)
    selection=choose()
    saved_selection=ROOT/"selection.json"
    if saved_selection.exists() and json.loads(saved_selection.read_text())!=selection:
        raise RuntimeError("Cached full-search results belong to a different screen selection")
    summary=ROOT/"cv-summary.json"
    digest=fingerprint()
    marker=ROOT/"split.sha256"
    if marker.exists() and marker.read_text().strip()!=digest:
        raise RuntimeError("Cached full-search results belong to a different split")
    if summary.exists():
        locked=json.loads(summary.read_text())
        if locked["split_sha256"]!=digest or locked["selection"]!=selection:
            raise RuntimeError("Existing CV summary disagrees with current selection/split")
        if "strong_fusion" in locked:
            print("Five-fold selection and strong fusion are already complete.",flush=True)
            return
    marker.write_text(digest+'\n')
    save_json(ROOT/"selection.json",selection)
    X=np.load(DATA/"dev_x.npy",mmap_mode="r")
    Y=np.load(DATA/"dev_y.npy",mmap_mode="r")
    latlon=np.load(DATA/"dev_latlon.npy",mmap_mode="r")
    folds=np.load(DATA/"folds.npy")
    rows=[outer_fold(selection,f,X,Y,latlon,t,v,i) for f,t,v,i in folds_of(folds)]
    ranking=[]
    for name in rows[0]:
        parts=[s[name] for s in rows]
        n=sum(p["n"] for p in parts)
        ranking.append({"name":name,"cv_mae_km":sum(p["mae_km"]*p["n"] for p in parts)/n,
                        "cv_rmse_km":(sum(p["mse_km2"]*p["n"] for p in parts)/n)**.5,
                        "fold_rmse_km":[p["rmse_km"] for p in parts]})
    ranking.sort(key=lambda x:x["cv_rmse_km"])
    save_json(ROOT/"cv-summary.json",{"winner":ranking[0]["name"],"candidates":ranking,
                                      "split_sha256":fingerprint(),"selection":selection})
    print(json.dumps(ranking[:10],indent=2),flush=True)


if __name__=="__main__":
    main()
