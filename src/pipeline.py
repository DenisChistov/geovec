from pathlib import Path
import os
import subprocess
import sys
import time

source = Path(__file__).resolve().parent
root = Path(os.environ.get('GEO_CV_ROOT', '/content/train250k-cv'))
until = time.monotonic() + 900
while not (root / 'split.json').exists() and time.monotonic() < until:
    time.sleep(5)
if not (root / 'split.json').exists():
    raise SystemExit('Data preparation did not finish. Check prepare.log.')
for phase in sys.argv[1:]:
    print(f'PHASE {phase}', flush=True)
    result = subprocess.run([sys.executable, '-u', str(source/'experiment.py'), phase], cwd=root)
    if result.returncode:
        print(f'FAILED {phase}: {result.returncode}', flush=True)
    else:
        print(f'COMPLETE {phase}', flush=True)
