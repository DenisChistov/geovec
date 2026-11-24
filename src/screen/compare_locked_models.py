"""Post-lock aggregate error profile for the two already scored models; never fits/selects."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from pyproj import Geod

data=Path(os.environ.get('GEO_CV_ROOT','/content/train250k-cv'))
sys.path.extend([str(data),str(Path(__file__).resolve().parent.parent)])
from experiment import decode,save_json

roots={'RMSE':data/'full500','MAE':data/'mae_cv'}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

scores={}
for k,root in roots.items():
    report=json.loads((root/'results.json').read_text())
    lock=json.loads((root/'cv-summary.json').read_text())
    if report['winner']!=lock['winner'] or report['cv_sha256']!=sha(root/'cv-summary.json') or report['predictions_sha256']!=sha(root/'test-predictions.npz'):
        raise ValueError('Evaluation is not tied to the immutable prediction set')
    scores[k]=report
assert scores['RMSE']['split_sha256']==scores['MAE']['split_sha256']
truth=np.load(data/'test_latlon.npy',mmap_mode='r',allow_pickle=False)
geod=Geod(ellps='WGS84');d={}
for k,root in roots.items():
    with np.load(root/'test-predictions.npz',allow_pickle=False) as ar:pred=ar['xyz']
    lat,lon=decode(pred);d[k]=geod.inv(truth[:,1],truth[:,0],lon,lat)[2]*.001
    assert len(d[k])==37500 and abs(d[k].mean()-scores[k]['test']['mae_km'])<1e-5
    assert abs(np.sqrt(np.mean(d[k]**2))-scores[k]['test']['rmse_km'])<1e-5
cuts=[1,5,10,25,50,100,250,500,750,1000,1500,2000,3000,5000,10000,20000]
qs=[.1,.25,.5,.75,.9,.95,.99]
profile={}
for k,v in d.items():
    profile[k]={'threshold_accuracy_percent':{str(t):float(100*(v<=t).mean()) for t in cuts},
                'quantile_km':{str(p):float(np.quantile(v,p)) for p in qs},
                'mse_share_above_1000km':float(np.sum(v[v>1000]**2)/np.sum(v**2)),
                'mse_share_above_3000km':float(np.sum(v[v>3000]**2)/np.sum(v**2))}
delta=d['MAE']-d['RMSE']
out={'n':len(delta),'split_sha256':scores['MAE']['split_sha256'],'models':profile,
     'paired_mae_model_closer_percent':float(100*(delta<0).mean()),
     'paired_rmse_model_closer_percent':float(100*(delta>0).mean()),
     'paired_tie_percent':float(100*(delta==0).mean()),
     'test_metrics_already_fixed':True,'note':'Post-lock aggregate analysis only. No selection, fitting, raw coordinates or row-level distances are exported.'}
save_json(data/'mae_cv'/'error-profile.json',out);print(json.dumps(out,indent=2),flush=True)
