"""Fit only the already selected MAE ensemble and evaluate historical holdout once."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import mae_search as search
import mae_rules,mae_gate,tune_rules,diversity_cv
from experiment import metrics,save_json
from predict_mae import query_predict,predict_xyz,save_robust,sha


def atomic(path,**values):
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**values);tmp.replace(path)


def main():
    data,out=search.DATA,search.OUT
    lock_path=out/'cv-summary.json';lock=json.loads(lock_path.read_text());cv_sha=sha(lock_path)
    rmse,rmse_sha=search.rmse_lock()
    if lock['rmse_cv_sha256']!=rmse_sha or lock['split_sha256']!=search.fingerprint():raise ValueError('MAE and RMSE split/lock mismatch')
    winner=lock['winner'];spec=lock['specifications'][winner]
    assert winner==min(lock['candidates'],key=lambda r:(r['cv_mae_km'],r['cv_rmse_km'],r['name']))['name']
    prediction_path=out/'test-predictions.npz';provenance_path=out/'final-provenance.json';result_path=out/'results.json'
    if prediction_path.exists():
        with np.load(prediction_path,allow_pickle=False) as ar:
            if str(ar['cv_sha256'])!=cv_sha or str(ar['winner'])!=winner:raise ValueError('Existing MAE prediction belongs to another CV lock')
    else:
        if result_path.exists():raise ValueError('Scored predictions missing')
        t0=perf_counter();state=None;checkpoint=None
        print('Refitting locked MAE winner:',winner,flush=True)
        if spec['kind'] in ('robust','gate'):
            Y=np.load(data/'dev_y.npy',mmap_mode='r',allow_pickle=False)
            folds=np.load(data/'folds.npy',allow_pickle=False)
            assert len(Y)==len(folds)==212500
            base=diversity_cv.final_oof(lock['members'],folds)
            if spec['kind']=='robust':
                if spec['pool']=='median':
                    _,_,base=search.build_final_dev(base,folds)
                state=mae_rules.fit_one(spec['parameters'],tune_rules.geometry(base,Y))
                checkpoint='mae-rule.npz';save_robust(out/checkpoint,state)
            else:
                x,f,_=search.build_final_dev(base,folds)
                state=mae_gate.fit_one(spec['parameters'],x,Y,search.gate_features(f),seed=search.SEED+5)
                checkpoint='mae-gate.pt';mae_gate.save(out/checkpoint,state,members=[*lock['members'],*[f'v3/expert-{j}' for j in range(7)],*lock['posterior_gate_experts']])
        Xtest=np.load(data/'test_x.npy',mmap_mode='r',allow_pickle=False)
        pred=query_predict(Xtest,data,spec,lock['members'],lock['posterior_gate_experts'],state)
        assert pred.shape==(37500,3) and np.isfinite(pred).all()
        assert search.rmse_lock()[1]==rmse_sha and sha(lock_path)==cv_sha
        prov={'winner':winner,'cv_sha256':cv_sha,'rmse_cv_sha256':rmse_sha,'split_sha256':lock['split_sha256'],
              'specification':spec,'checkpoint':checkpoint,'checkpoint_sha256':sha(out/checkpoint) if checkpoint else None,
              'development_rows':212500,'test_rows':37500,'shared_experts':'V3/V4 and frozen RMSE final constituent checkpoints already fitted on all development rows',
              'fit_objective':'MAE-aligned robust/decision rule; select by exact WGS84 MAE',
              'fit_seconds':perf_counter()-t0}
        save_json(provenance_path,prov)
        atomic(prediction_path,xyz=pred,cv_sha256=cv_sha,winner=winner)
    pred_sha=sha(prediction_path)
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['predictions_sha256']!=pred_sha or old['cv_sha256']!=cv_sha:raise ValueError('MAE test score differs from committed predictions')
        print(json.dumps(old,indent=2),flush=True);return
    # The holdout labels are opened only after winner and all predictions are committed.
    with np.load(prediction_path,allow_pickle=False) as ar:pred=ar['xyz']
    truth=np.load(data/'test_latlon.npy',mmap_mode='r',allow_pickle=False)
    old_rmse=json.loads((search.RMSE/'results.json').read_text())
    v4_path=data/'v4'/'results.json'
    old_v4=json.loads(v4_path.read_text())['test'] if v4_path.exists() else old_rmse['historical_reference']
    result={'winner':winner,'cv':lock['candidates'][0],'test':metrics(pred,truth),'test_rows':37500,'dev_rows':212500,
            'cv_sha256':cv_sha,'rmse_cv_sha256':rmse_sha,'split_sha256':lock['split_sha256'],'predictions_sha256':pred_sha,
            'historical_rmse_model':old_rmse['test'],
            'historical_v4':old_v4,
            'test_note':'Same historical aggregate results known from earlier experiments; no test metrics used to select new MAE model'}
    save_json(result_path,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
