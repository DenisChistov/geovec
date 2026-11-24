"""Post-lock diagnostics; never alters the selected extra experts."""
from pathlib import Path
import os
import sys
import numpy as np

from diversity_select import read,write,digest,screen_prior,SOURCES
from diversity_context import supplement

data=Path(os.environ.get('GEO_CV_ROOT','/content/train250k-cv'))
screen=data/'screen500';dest=data/'diversity500'
sys.path.extend([str(data),str(Path(__file__).resolve().parent.parent)])
lock=read(dest/'selection.json')
assert lock['split_sha256']==digest(screen/'split.npz')
with np.load(screen/'split.npz',allow_pickle=False) as split:
    fit,cal,evaluate=[split[k] for k in ('fit','cal','evaluate')]
folds=np.load(data/'folds.npy');train=np.flatnonzero(folds!=0);val=np.flatnonzero(folds==0)
y=np.load(data/'dev_y.npy',mmap_mode='r')
names=sorted(k for k,v in read(screen/'registry.json').items() if v['family'] in SOURCES)
prior=screen_prior(data,train,val,folds,fit,cal,evaluate,y[fit])
write(dest/'supplement.json', supplement(data,screen,names,cal,evaluate,val,prior[:len(cal)],prior[len(cal):],y[cal],y[evaluate]))
print('Supplement completed',flush=True)
