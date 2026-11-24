"""Export compact coefficient metadata for documentation; no row-level inputs."""
from pathlib import Path
import json
import numpy as np

root=Path('/content/train250k-cv/full500')
lock=json.loads((root/'cv-summary.json').read_text())
prov=json.loads((root/'final-provenance.json').read_text())
model=root/prov['models_relative_to_full500']
winner=lock['winner']
info={'winner':winner,'plan':prov['plan'],'component_training_seeds':prov['component_training_seeds'],
      'parameters':None,'small_fusion_checkpoint':{}}
for plan in ('tuned_fusion','diversity_fusion','strong_fusion'):
    if winner in lock.get(plan,{}).get('rules',{}):
        info['parameters']=lock[plan]['rules'][winner]
        info['members']=lock[plan]['members']
for name in ('tuned-rule.npz','diversity-rule.npz','strong-rule.npz','fusion-rule.npz'):
    path=model/name
    if path.exists():
        with np.load(path,allow_pickle=False) as ar:
            for k in ar.files:
                v=ar[k]
                # Keep learned coefficients/normalization and small configuration
                # values. Never export row-level reference coordinates or errors.
                if v.size<=512 and not any(word in k.lower() for word in ('row','truth','error','anchor','keys','reference','train')):
                    info['small_fusion_checkpoint'][k]=v.tolist()
                else:
                    info['small_fusion_checkpoint'][k]={'shape':list(v.shape),'omitted':True}
(root/'winner-parameters.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in info.items() if k!='small_fusion_checkpoint'},ensure_ascii=False,indent=2),flush=True)
