from pathlib import Path
import json
import os
import subprocess

p = Path(os.environ.get('GEO_CV_ROOT', '/content/train250k-cv'))
log_file = p / 'pipeline.log'
log = log_file.read_text().splitlines() if log_file.exists() else []
for phase in ('catboost','dnn','siamese','neighbors'):
    files = list(p.glob(f'cv-{phase}-[0-4]-*.json'))
    summary = p/f'cv-{phase}-summary.json'
    row = json.loads(summary.read_text())[0] if summary.is_file() else None
    print(phase, 'fits', len(files), f"CV RMSE={row['rmse_km']:.1f}" if row else '', flush=True)
for l in log[-6:]:
    print(l[:750], flush=True)
if (p/'results.json').is_file():
    print('RESULTS_READY', flush=True)
pid_file = p / 'pipeline.pid'
if pid_file.exists():
    state = pid_file.read_text().strip()
    print('Pipeline:', subprocess.run(['ps','-p',state,'-o','stat=,etime='], capture_output=True, text=True).stdout.strip(), flush=True)
