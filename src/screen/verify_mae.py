"""Replay the new MAE winner from saved weights, without opening target labels."""
import json
import os
from pathlib import Path
import sys
import numpy as np
from pyproj import Geod

data=Path(os.environ.get('GEO_CV_ROOT','/content/train250k-cv'))
sys.path.extend([str(data),str(data/'screen-src')])
from predict_mae import predict_xyz
from experiment import decode,save_json

root=data/'mae_cv'
x=np.load(data/'test_x.npy',mmap_mode='r',allow_pickle=False)[:4096]
with np.load(root/'test-predictions.npz',allow_pickle=False) as ar:expected=ar['xyz'][:len(x)]
answer=predict_xyz(x,data);lat,lon=decode(answer);a,b=decode(expected)
d=Geod(ellps='WGS84').inv(lon,lat,b,a)[2]*.001
report={'n':len(x),'median_km':float(np.median(d)),'p95_km':float(np.quantile(d,.95)),
        'max_km':float(d.max()),'mean_km':float(d.mean()),'test_truth_opened':False}
save_json(root/'replay-check.json',report);print(json.dumps(report,indent=2),flush=True)
