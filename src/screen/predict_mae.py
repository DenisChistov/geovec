"""Inference for the independent MAE winner; the RMSE model remains untouched."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from predict_final import _Inference,PRIOR,PAPER
from experiment import decode
from run_v3 import blend,unit
import mae_posterior
import mae_rules
import mae_gate
import tune_rules


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def base_outputs(inf,members):
    ans={}
    raw=[k for k in members if k.startswith(('neural/','geo/','retrieval/'))]
    if raw:inf._raw(raw,ans)
    if PRIOR in members:ans[PRIOR]=inf._v4()
    if PAPER in members:ans[PAPER]=inf._paper('distance_risk','paper-distance-risk.pt')
    meta=[k for k in members if k.startswith('meta/')]
    if meta:inf._meta(meta,ans)
    if set(ans)!=set(members):raise ValueError('Missing MAE shared expert predictions')
    return np.stack([ans[k] for k in members],axis=1)


def save_robust(path,state):
    fields={k:np.asarray(v) for k,v in state.items()}
    if any(v.dtype.hasobject for v in fields.values()):raise ValueError('Robust rule must have numeric or string state')
    tmp=path.with_suffix('.tmp.npz');np.savez(tmp,**fields);tmp.replace(path)


def load_robust(path):
    with np.load(path,allow_pickle=False) as s:
        return {k:(s[k].item() if s[k].ndim==0 else s[k]) for k in s.files}


def query_predict(X,reference,spec,members,posterior_ids,state=None,full_dir=None):
    if not len(X):return np.empty((0,3),np.float32)
    inf=_Inference(X,reference,full_dir)
    kind=spec['kind']
    if kind=='posterior':
        return unit(mae_posterior.predictions(inf.bundle,only=[spec['method']])[spec['method']])
    if kind=='blend':
        p=mae_posterior.predictions(inf.bundle,only=[spec['posterior']])[spec['posterior']]
        v=base_outputs(inf,[spec['expert']])[:,0]
        return blend(p,v,spec['weight'])
    base=base_outputs(inf,members)
    if kind=='robust':
        if spec['pool']=='median':
            p=mae_posterior.predictions(inf.bundle,only=[posterior_ids[0]])[posterior_ids[0]]
            base=np.concatenate([p[:,None,:],base],axis=1)
        return unit(mae_rules.predict_geometry(tune_rules.geometry(base),state))
    if kind=='gate':
        b=inf.bundle
        post=mae_posterior.predictions(b,only=posterior_ids)
        directions=np.concatenate([base,b['experts'],np.stack([post[k] for k in posterior_ids],axis=1)],axis=1)
        return unit(mae_gate.predict(state,directions,{'v3_features':b['features']}))
    raise ValueError('Unknown MAE model kind')


def predict_xyz(X,reference,mae_dir=None):
    reference=Path(reference);root=Path(mae_dir or reference/'mae_cv')
    lock_path=root/'cv-summary.json';lock=json.loads(lock_path.read_text())
    p=json.loads((root/'final-provenance.json').read_text())
    if p['winner']!=lock['winner'] or p['cv_sha256']!=sha(lock_path):raise ValueError('MAE model and CV lock disagree')
    if p['rmse_cv_sha256']!=sha(reference/'full500'/'cv-summary.json'):raise ValueError('Shared RMSE models changed')
    spec=lock['specifications'][lock['winner']];state=None
    if p.get('checkpoint'):
        path=root/p['checkpoint']
        if sha(path)!=p['checkpoint_sha256']:raise ValueError('MAE checkpoint differs from recorded fit')
        state=load_robust(path) if spec['kind']=='robust' else mae_gate.load(path)
    return query_predict(X,reference,spec,lock['members'],lock['posterior_gate_experts'],state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path(os.environ.get('GEO_CV_ROOT','/content/train250k-cv')))
    args=p.parse_args()
    pred=predict_xyz(np.load(args.input,mmap_mode='r',allow_pickle=False),args.reference)
    ll=np.column_stack(decode(pred))
    if args.output.suffix=='.npy':np.save(args.output,ll,allow_pickle=False)
    elif args.output.suffix=='.csv':np.savetxt(args.output,ll,delimiter=',',header='latitude,longitude',comments='')
    else:raise ValueError('Output must be .npy or .csv')
    print(f'Saved {len(pred)} WGS84 coordinates',flush=True)


if __name__=='__main__':main()
