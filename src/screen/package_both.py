"""Combine the original RMSE package and MAE checkpoints; no row-level data."""
from pathlib import Path
import json
from zipfile import ZipFile,ZIP_DEFLATED

data=Path('/content/train250k-cv');mae=data/'mae_cv'
source=Path('/content/train250k-final-model.zip');target=Path('/content/train250k-models-mae-rmse.zip')
prov=json.loads((mae/'final-provenance.json').read_text())
names=['cv-summary.json','results.json','final-provenance.json','replay-check.json']
if prov['checkpoint']:names.append(prov['checkpoint'])
with ZipFile(source,'r') as previous,ZipFile(target,'w',ZIP_DEFLATED,compresslevel=4) as out:
    for item in previous.infolist():
        if item.filename!='README.txt':out.writestr(item.filename,previous.read(item))
    for name in names:
        if (mae/name).is_file():out.write(mae/name,'mae_cv/'+name)
    out.writestr('README.txt','Both CV-locked models. Extract into GEO_CV_ROOT (same frozen split). Use src/screen/predict_final.py for RMSE and src/screen/predict_mae.py for MAE. The original training bank must be recreated separately. No training/test datasets or per-row forecasts included.\n')
print(target,target.stat().st_size,'bytes')
