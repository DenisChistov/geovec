"""Predict new WGS84 coordinates with the V4 cross-validation winner."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiment import decode
from predict_v3 import normalize_features, build_bundle, _project_saved, predict_xyz as predict_v3
from run_v3 import take_features, unit
from run_v4 import registry


def predict_xyz(X, models, base_models, reference):
    values=normalize_features(X)
    if not len(values):
        return np.empty((0,3),dtype=np.float32)
    models=Path(models)
    winner=json.loads((models/"cv-summary.json").read_text())["winner"]
    if winner=="incumbent_v3":
        return predict_v3(values,base_models,reference)
    module,kind,_=registry()[winner]
    bundle=build_bundle(values,base_models,reference)
    if kind=="deterministic":
        return unit(module.predict(bundle)[winner])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reduced=_project_saved(models/"best-pca.npz",values,device)
    features=take_features(bundle,reduced)
    return unit(module.predict_saved(features,bundle["experts"],models/f"{winner}.npz"))


def main():
    cmd=argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("--input",type=Path,required=True,help="NumPy array of 1280-dimensional vectors")
    cmd.add_argument("--output",type=Path,required=True)
    cmd.add_argument("--reference",type=Path,default=Path("/content/train250k-cv"))
    cmd.add_argument("--models",type=Path)
    cmd.add_argument("--base-models",type=Path)
    args=cmd.parse_args()
    xyz=predict_xyz(np.load(args.input,allow_pickle=False,mmap_mode="r"),
                    args.models or args.reference/"v4",
                    args.base_models or args.reference/"v3",args.reference)
    lat,lon=decode(xyz)
    np.savetxt(args.output,np.column_stack((lat,lon)),delimiter=",",header="latitude,longitude",
               comments="",fmt="%.10f")
    print(f"Saved {len(xyz)} predictions")


if __name__=="__main__":
    main()
