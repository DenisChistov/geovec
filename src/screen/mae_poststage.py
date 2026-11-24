"""Cache the label-free posterior rules and development-only metrics."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import full_search
import mae_posterior
from experiment import metrics,save_json
from run_v3 import fingerprint,folds_of,unit
from run_v4 import load_bundle

DATA=full_search.DATA
OUT=DATA/'mae_cv'/'posterior-cache'


def _signature(key):
    path=DATA/'v3'/f'bundle-{key}.npz'
    st=path.stat()
    h=hashlib.sha256(json.dumps([key,fingerprint(),st.st_size,st.st_mtime_ns,
                mae_posterior.grid(),list(mae_posterior.GATE_EXPERTS)],sort_keys=True).encode())
    h.update(Path(mae_posterior.__file__).read_bytes());return h.hexdigest()


def cache(key,bundle,truth=None):
    OUT.mkdir(parents=True,exist_ok=True)
    sig=_signature(key);p=OUT/f'{key}.npz';s=OUT/f'{key}.json'
    if p.exists() and s.exists():
        detail=json.loads(s.read_text())
        if detail['signature']==sig and (truth is None or set(detail['scores'])==set(mae_posterior.grid())):
            with np.load(p,allow_pickle=False) as ar:
                if str(ar['signature'])==sig:
                    pred={k:ar[k] for k in mae_posterior.GATE_EXPERTS}
                    if all(v.shape==(len(bundle['features']),3) and np.isfinite(v).all() for v in pred.values()):
                        return pred,detail['scores']
    raw=mae_posterior.predictions(bundle,only=None if truth is not None else mae_posterior.GATE_EXPERTS)
    scores={k:metrics(unit(v),truth) for k,v in raw.items()} if truth is not None else {}
    pred={k:raw[k] for k in mae_posterior.GATE_EXPERTS}
    temp=p.with_suffix('.tmp.npz');np.savez(temp,signature=sig,**pred);temp.replace(p)
    save_json(s,{'signature':sig,'scores':scores});return pred,scores


def main():
    ll=np.load(DATA/'dev_latlon.npy',mmap_mode='r');folds=np.load(DATA/'folds.npy')
    for f,t,v,inner in folds_of(folds):
        t0=perf_counter();_,scores=cache(f'o{f}',load_bundle(f'o{f}'),ll[v])
        print('Posterior fold',f+1,'/5',sorted([(k,round(s['mae_km'],2)) for k,s in scores.items()],key=lambda x:x[1])[:5],flush=True)
        for j,(_,q) in enumerate(inner):cache(f'o{f}-i{j}',load_bundle(f'o{f}-i{j}'))
        print('Posterior cache fold',f+1,'/5',round(perf_counter()-t0,1),'s',flush=True)
    (OUT/'COMPLETE').write_text(fingerprint()+'\n')


if __name__=='__main__':main()
