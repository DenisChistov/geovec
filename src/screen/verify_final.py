"""Replay a committed final model on features, without reading test truth."""
from pathlib import Path
import json
import os
import sys
import numpy as np
from pyproj import Geod

data=Path(os.environ.get('GEO_CV_ROOT','/content/train250k-cv'))
sys.path.extend([str(data),str(data/'screen-src')])
from predict_final import predict_xyz
from experiment import decode,save_json

root=data/'full500'
x=np.load(data/'test_x.npy',mmap_mode='r',allow_pickle=False)[:4096]
with np.load(root/'test-predictions.npz',allow_pickle=False) as saved:
    expected=saved['xyz'][:len(x)]
p= predict_xyz(x,data)
lat,lon=decode(p);a,b=decode(expected)
distance=Geod(ellps='WGS84').inv(lon,lat,b,a)[2]*.001
report={'n':len(x),'median_km':float(np.median(distance)), 'p95_km':float(np.quantile(distance,.95)),
        'max_km':float(distance.max()), 'mean_km':float(distance.mean()),
        'metric':'WGS84 between checkpoint replay and committed predictions, not ground truth',
        'test_truth_opened':False}
save_json(root/'replay-check.json',report)
print(json.dumps(report,indent=2),flush=True)
