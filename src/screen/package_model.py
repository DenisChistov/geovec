"""Package saved checkpoints and aggregate metadata without input banks or predictions."""
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

root=Path('/content/train250k-cv')
full=root/'full500'
prov=json.loads((full/'final-provenance.json').read_text())
model=full/prov['models_relative_to_full500']
paths=[full/x for x in ('cv-summary.json','final-provenance.json','results.json','replay-check.json','winner-parameters.json') if (full/x).is_file()]
paths.extend(p for p in model.rglob('*') if p.is_file())
# These are the deployed, all-development base models and V4 calibrator.
# Do not include saved real coordinates, training arrays, or per-row predictions.
for folder,names in [('v3',('cells-512-a.pt','cells-1024.pt','cells-512-b.pt','direct.pt')),
                     ('v4',('cal_interaction.npz','best-pca.npz','cv-summary.json'))]:
    paths.extend(root/folder/n for n in names if (root/folder/n).is_file())
target=Path('/content/train250k-final-model.zip')
with ZipFile(target,'w',ZIP_DEFLATED,compresslevel=4) as archive:
    for path in paths:archive.write(path,path.relative_to(root))
    archive.writestr('README.txt','Unzip into GEO_CV_ROOT (matching the frozen split). The original development feature & coordinate bank must be reconstructed separately with src/prepare.py. No training data or holdout predictions/labels are included in this archive. For inference see src/screen/predict_final.py in project archive.\n')
print(target, target.stat().st_size, 'bytes',len(paths),'files')
