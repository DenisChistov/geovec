from collections import Counter
from pathlib import Path
import json
import os

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(os.environ.get('GEO_CV_ROOT', '/content/train250k-cv'))
SEED = 417359
COLS = ['id', 'shard', 'latitude', 'longitude', 'country']

def find_folder():
    folder = Path(os.environ.get('TRAIN_250K_DIR', '/content/drive/MyDrive/train_250k'))
    if not (folder / 'train_manifest.parquet').is_file():
        raise FileNotFoundError(f'Set TRAIN_250K_DIR to the folder containing train_manifest.parquet: {folder}')
    return folder


def encode(latitude, longitude):
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)
    # Unit direction of ellipsoidal ECEF position at zero altitude, WGS84.
    e2 = 1 / 298.257223563 * (2 - 1 / 298.257223563)
    y = np.column_stack([np.cos(lat) * np.cos(lon),
                         np.cos(lat) * np.sin(lon),
                         (1 - e2) * np.sin(lat)])
    return (y / np.linalg.norm(y, axis=1, keepdims=True)).astype(np.float32)

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'split.json').exists():
        print('Existing frozen splits and feature arrays found.', flush=True)
        return
    folder = find_folder()
    df = pq.read_table(folder / 'train_manifest.parquet', columns=COLS).to_pandas()
    sz = len(df)
    assert sz == 250_000 and df.id.is_unique
    ids = df.id.to_numpy()
    lat = df.latitude.to_numpy()
    lon = df.longitude.to_numpy()
    names = df.country.fillna('').to_numpy()
    assert np.isfinite(lat).all() and np.isfinite(lon).all()
    print(f'Manifest verified, n={sz}', flush=True)

    all_features = np.empty((sz, 1280), dtype=np.float32)
    for count, (shard, group) in enumerate(df.groupby('shard', sort=True), start=1):
        stem = Path(shard).stem
        indices = group.index.to_numpy()
        meta = pq.read_table(folder / f'{stem}_metadata.parquet', columns=COLS).to_pandas()
        assert np.array_equal(meta.id.to_numpy(), ids[indices])
        assert np.array_equal(meta.latitude.to_numpy(), lat[indices])
        assert np.array_equal(meta.longitude.to_numpy(), lon[indices])
        embeddings = load_file(str(folder / f'{stem}_features.safetensors'))
        for name, start, end in [('geoclip', 0, 512), ('streetclip', 512, 1280)]:
            vectors = embeddings[name].astype(np.float32)
            assert vectors.shape == (len(indices), end - start) and np.isfinite(vectors).all()
            lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
            assert np.all(lengths > 0)
            all_features[indices, start:end] = vectors / lengths
        if count % 10 == 0:
            print(f'Verified and loaded {count} shards', flush=True)
    assert count == 89
    counts = Counter(names)
    strata = np.where(np.array([counts[c] >= 20 for c in names]), names, 'OTHER')
    dev, test = train_test_split(np.arange(sz), test_size=37_500, random_state=SEED, stratify=strata)
    dev, test = np.sort(dev), np.sort(test)
    fold_ids = np.full(len(dev), -1, dtype=np.int8)
    for fold, (_, val) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(dev, strata[dev])):
        fold_ids[val] = fold
    assert len(dev) == 212500 and len(test) == 37500 and np.bincount(fold_ids).tolist() == [42500]*5
    assert not np.intersect1d(dev, test).size
    for split, rows in [('dev', dev), ('test', test)]:
        np.save(ROOT / f'{split}_x.npy', all_features[rows])
        np.save(ROOT / f'{split}_y.npy', encode(lat[rows], lon[rows]))
        np.save(ROOT / f'{split}_latlon.npy', np.stack([lat[rows], lon[rows]], axis=1))
        np.save(ROOT / f'{split}_rows.npy', rows)
    np.save(ROOT / 'folds.npy', fold_ids)
    spec = {'seed': SEED, 'trainval': len(dev), 'test': len(test), 'fold_sizes': np.bincount(fold_ids).tolist(),
            'source': 'train_manifest.parquet', 'label': 'unit WGS84 ECEF direction', 'shards': count}
    stage = ROOT / 'split.tmp'
    stage.write_text(json.dumps(spec, indent=2))
    stage.replace(ROOT / 'split.json')
    print(spec, flush=True)

if __name__ == '__main__':
    main()
