"""Literature-inspired geolocation models using frozen image embeddings."""
from math import pi, sqrt

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from pyproj import Geod

from cell_v2 import _spherical_cells, _features, _spherical_mean
from refine_v3 import _spherical_mean as row_mean

DEVICE = "cuda"
GEOD = Geod(ellps="WGS84")

METHODS = {
    "distance_smoothing": "PIGEON-inspired geographically smoothed cross-entropy",
    "distance_risk": "Distance-smoothed cross-entropy plus expected squared angular risk",
    "hierarchical_cells": "PlaNet-inspired balanced geographic tree with conditional coarse/fine classification",
    "fourier_location": "GeoCLIP-inspired continuous Fourier GPS encoder and image-to-location similarity",
    "candidate_ranker": "GeoRanker-inspired listwise candidate scoring with WGS84 distance supervision",
}


def balanced_cells(x):
    """A deterministic median-split tree: 64 parent nodes, 512 final cells."""
    groups = [np.arange(len(x))]
    for level in range(9):
        children = []
        for ids in groups:
            a = x[ids]
            centered = a-a.mean(0)
            _,vectors = np.linalg.eigh(centered.T @ centered)
            principal = vectors[:, -1]
            order = ids[np.argsort(a @ principal,kind="stable")]
            midpoint = len(order)//2
            children.extend([order[:midpoint],order[midpoint:]])
        groups=children
    labels=np.empty(len(x),dtype=np.int64)
    centers=[]
    for i,group in enumerate(groups):
        labels[group]=i
        direction=x[group].mean(axis=0)
        centers.append(direction/np.linalg.norm(direction))
    return torch.as_tensor(np.array(centers),device=DEVICE),torch.as_tensor(labels,device=DEVICE)


class GeographicClassifier(nn.Module):
    def __init__(self,k,hierarchy=False):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(1280,640),nn.LayerNorm(640),nn.GELU(),
                                    nn.Dropout(.12),nn.Linear(640,384),nn.GELU(),nn.Dropout(.12))
        self.fine=nn.Linear(384,k)
        self.coarse=nn.Linear(384,64) if hierarchy else None

    def forward(self,image):
        x=self.backbone(image)
        return self.fine(x), self.coarse(x) if self.coarse is not None else None


class ContinuousLocator(nn.Module):
    """Two encoders, with fixed multiresolution Fourier features for GPS."""
    def __init__(self,centers,seed):
        super().__init__()
        gen=torch.Generator().manual_seed(seed)
        frequencies=torch.cat([torch.randn(3,24,generator=gen)*s for s in (1,4,16,64)],dim=1)
        waves=2*pi*centers.cpu()@frequencies
        self.register_buffer("location_features",torch.cat([centers.cpu(),waves.sin(),waves.cos()],dim=1))
        self.register_buffer("fourier_frequencies",frequencies)
        self.image=nn.Sequential(nn.Linear(1280,640),nn.LayerNorm(640),nn.GELU(),nn.Dropout(.12),
                                 nn.Linear(640,256),nn.GELU(),nn.Linear(256,128))
        self.location=nn.Sequential(nn.Linear(self.location_features.shape[1],256),nn.GELU(),nn.Linear(256,128))
        self.scale=nn.Parameter(torch.tensor(3.5))

    def forward(self,image):
        a=F.normalize(self.image(image),dim=1)
        b=F.normalize(self.location(self.location_features),dim=1)
        return self.scale.clamp(0,4.5).exp()*(a@b.T),None


def probabilities(logits,coarse):
    if coarse is None:
        return logits.softmax(dim=1)
    return (logits.reshape(-1,64,8).softmax(dim=2)*coarse.softmax(dim=1)[:,:,None]).flatten(1)


def fit_cells(method,X,Y,train,query,seed,save_path=None):
    torch.manual_seed(seed)
    rng=torch.Generator(device=DEVICE).manual_seed(seed)
    points=torch.as_tensor(Y[train],device=DEVICE)
    hierarchy=method=="hierarchical_cells"
    continuous=method=="fourier_location"
    k=1024 if continuous else 512
    if hierarchy:
        centers,labels=balanced_cells(Y[train])
    else:
        centers,labels,_=_spherical_cells(points,k,rng)
    model=(ContinuousLocator(centers,seed) if continuous else GeographicClassifier(k,hierarchy)).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.002,fused=True)
    epochs=20 if continuous else 17
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00012)
    scaler=torch.amp.GradScaler("cuda")
    features=_features(X,train,torch.device(DEVICE))
    for _ in range(epochs):
        for idx in torch.randperm(len(points),device=DEVICE,generator=rng).split(4096):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                fine,coarse=model(features[idx])
            fine=fine.float()
            # The original PIGEON applies a haversine-based smoothing. Use a
            # normalized target and central angle on WGS84 ECEF directions.
            dot=(points[idx]@centers.T).clamp(-1,1)
            distance=torch.acos(dot)*6371.0088
            tau=150. if method=="distance_risk" else 75.
            soft=(-distance/tau).softmax(dim=1)
            loss=-(soft*fine.log_softmax(dim=1)).sum(dim=1).mean()
            if hierarchy:
                loss+=.35*F.cross_entropy(coarse.float(),labels[idx]//8)
            if method=="distance_risk":
                loss+=.20*(fine.softmax(dim=1)*(distance/4000).square()).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        schedule.step()
    if save_path is not None:
        torch.save({"method":method,"seed":seed,"centers":centers.cpu(),
                    "state_dict":{k:v.cpu() for k,v in model.state_dict().items()}},save_path)
    model.eval()
    answer=[]
    with torch.inference_mode():
        for batch in _features(X,query,torch.device(DEVICE)).split(4096):
            with torch.autocast("cuda"):
                fine,coarse=model(batch)
            prob=probabilities(fine.float(),coarse.float() if coarse is not None else None)
            mode=centers[prob.argmax(dim=1)]
            answer.append(_spherical_mean(prob,centers,mode).cpu().numpy())
    return np.concatenate(answer)


def candidate_inputs(bundle):
    f=np.asarray(bundle["features"],dtype=np.float32)
    experts=np.asarray(bundle["experts"],dtype=np.float32)
    ranks=[f[:,i:i+40].reshape(-1,10,4) for i in (21,74,127)]
    points=np.concatenate([*(a[:,:,1:] for a in ranks),experts],axis=1)
    p=np.concatenate([*(a[:,:,0] for a in ranks),np.full((len(f),7),.1,dtype=np.float32)],axis=1)
    agreement=np.einsum("nkd,nmd->nkm",points,experts)
    source=np.arange(37)
    source=np.minimum(source//10,2).astype(int)
    source[30:]=np.arange(3,10)
    source=np.broadcast_to(np.eye(10,dtype=np.float32)[source],(len(f),37,10))
    context=np.column_stack([*[f[:,i+41] for i in (21,74,127)],
                             *[f[:,i+49] for i in (21,74,127)],f[:,180+5],f[:,180+7]])
    context=np.broadcast_to(context[:,None,:],(len(f),37,context.shape[1]))
    feat=np.concatenate([points, np.log(np.maximum(p,1e-6))[...,None],
                         agreement,source,context],axis=2).astype(np.float32)
    return points,feat


def geodetic(x):
    x,y,z=np.moveaxis(np.asarray(x,dtype=np.float64),-1,0)
    e2=1-(1-1/298.257223563)**2
    return np.degrees(np.arctan2(y,x)),np.degrees(np.arctan2(z,(1-e2)*np.hypot(x,y)))


def distances(points,truth):
    flat=[]
    for start in range(0,len(points),4096):
        poly=points[start:start+4096]
        q=truth[start:start+4096]
        a,b=geodetic(poly.reshape(-1,3))
        c,d=geodetic(np.repeat(q,poly.shape[1],axis=0))
        d_km=GEOD.inv(a,b,c,d)[2].reshape(poly.shape[:2])*.001
        flat.append(d_km.astype(np.float32))
    return np.concatenate(flat)


def fit_ranker(train_bundle,y,eval_bundle,seed,save_path=None):
    torch.manual_seed(seed)
    pts,features=candidate_inputs(train_bundle)
    valid_pts,valid_features=candidate_inputs(eval_bundle)
    d=torch.as_tensor(distances(pts,y),device=DEVICE)
    x=torch.as_tensor(features,device=DEVICE,dtype=torch.float16)
    model=nn.Sequential(nn.Linear(x.shape[2],64),nn.LayerNorm(64),nn.GELU(),
                        nn.Linear(64,32),nn.GELU(),nn.Linear(32,1)).to(DEVICE)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.003,fused=True)
    scaler=torch.amp.GradScaler("cuda")
    for _ in range(8):
        for batch in torch.randperm(len(x),device=DEVICE).split(2048):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                estimated=model(x[batch]).squeeze(-1).float()
            true=d[batch]
            ranking=(-true/500).softmax(dim=1)
            loss=F.mse_loss(estimated,torch.log1p(true/250))
            loss+=.35*-(ranking*(-estimated/.35).log_softmax(dim=1)).sum(dim=1).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    if save_path is not None:
        torch.save(model.state_dict(),save_path)
    model.eval()
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(valid_pts),2048):
            section=slice(start,start+2048)
            feats=torch.as_tensor(valid_features[section],device=DEVICE,dtype=torch.float16)
            with torch.autocast("cuda"):
                risk=model(feats).squeeze(-1).float()
            weight=(-risk/.35).softmax(dim=1)
            xyz,_=row_mean(torch.as_tensor(valid_pts[section],device=DEVICE),weight)
            predictions.append(xyz.cpu().numpy())
    return np.concatenate(predictions)
