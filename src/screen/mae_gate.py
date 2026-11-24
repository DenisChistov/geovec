"""Conditional geographic gates trained on caller-supplied out-of-fold experts.

Inputs are normalized WGS84 ECEF directions. A plain f matrix adds context.
With f={'v3_features': matrix}, the first four ranked cell centers of each V3
classifier become additional actions. The seven native V3 forecasts are added
only if the caller explicitly sets add_v3_experts=True. No query labels enter
prediction or feature normalization. Official model selection uses WGS84 MAE.
"""

import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VERSION = 1
RADIUS_KM = 6371.0088
E2 = 1 - (1 - 1 / 298.257223563) ** 2
RANKS = 4
RISK_KNOTS_KM = (20., 80., 220., 550., 1300., 3200., 8000., 15000.)
RECIPES = ("linear", "mean", "median", "residual", "risk", "oracle", "cdf", "laplace")
BATCH = 2048


def grid():
    """One spec per independently scored action rule. Training is shared by recipe."""
    entries = (
        ("linear", "linear", "mean", 1., "Линейный условный шлюз, оптимизированный по среднему угловому расстоянию."),
        ("mean", "mean", "mean", 1., "Нелинейные веса кандидатов и касательная комбинация с угловым L1-критерием."),
        ("sharpen", "mean", "mean", .52, "Тот же условный шлюз с более резким распределением весов при прогнозе."),
        ("median", "median", "median", 1., "Обучаемые веса и сглаженная геометрическая медиана в касательной плоскости."),
        ("median_sharp", "median", "median", .60, "Геометрическая медиана с усилением наиболее надёжных голосов."),
        ("residual", "residual", "residual", 1., "Медиана смеси с ограниченной обучаемой поправкой по востоку и северу."),
        ("risk_soft", "risk", "mean", 1., "Условный выигрыш по расстоянию каждого кандидата относительно V4 и мягкое решение."),
        ("risk_hard", "risk", "hard", 1., "Выбор одной исходной точки по предсказанному приращению среднего расстояния."),
        ("risk_median", "risk", "median", 1., "Геометрическая медиана с весами по условному среднему риску кандидатов."),
        ("oracle_mode", "oracle", "hard", 1., "Классификатор близких к истинной точке кандидатов с жёстким выбором."),
        ("oracle_median", "oracle", "median", 1., "Медиана из распределения вероятности стать близким географическим кандидатом."),
        ("cdf_soft", "cdf", "mean", 1., "Оценка ожидаемого расстояния интегрированием вероятностей превышения восьми порогов."),
        ("cdf_hard", "cdf", "hard", 1., "Одиночный кандидат по интегральной оценке распределения ошибки."),
        ("laplace_median", "laplace", "density", 1., "Смесь сферических ядер Лапласа с индивидуальной шириной и сглаженным медианным решением."),
        ("laplace_mode", "laplace", "mode", 1., "Выбор географической моды смеси ядер Лапласа по массе и концентрации."),
    )
    return {
        "mae_gate:"+key: dict(name="mae_gate:"+key, recipe=recipe, decode=decode,
                              temperature=temp, description=description)
        for key, recipe, decode, temp, description in entries
    }


def _spec(spec):
    if isinstance(spec, str):
        return grid()[spec]
    if not isinstance(spec, dict) or spec.get("name") not in grid() or spec != grid()[spec["name"]]:
        raise ValueError("Unrecognized geographic MAE gate specification")
    return dict(spec)


def _device(value):
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


def _matrix(values, name, rows=None, ndim=2):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != ndim or (rows is not None and len(arr) != rows):
        raise ValueError(f"{name} has incompatible dimensions")
    for start in range(0,len(arr),8192):
        if not np.isfinite(arr[start:start+8192]).all():
            raise ValueError(f"{name} contains nonfinite values")
    return arr


def _inputs(experts, f):
    e = _matrix(experts,"experts",ndim=3)
    if e.shape[2] != 3 or e.shape[1] == 0:
        raise ValueError("Experts must have shape [rows, positive expert count, 3]")
    if f is None:
        f = {}
    elif not isinstance(f,dict):
        f = {"context":f}
    unknown_keys = set(f) - {"v3_features","add_v3_experts","context","candidates"}
    if unknown_keys:
        raise ValueError(f"Unknown gate feature keys {sorted(unknown_keys)}")
    v3 = None if f.get("v3_features") is None else _matrix(f["v3_features"],"v3_features",len(e))
    if v3 is not None and v3.shape[1] < 246:
        raise ValueError("V3 features must start with the 246 original columns")
    context = None if f.get("context") is None else _matrix(f["context"],"context",len(e))
    extra = None if f.get("candidates") is None else _matrix(f["candidates"],"candidates",len(e),ndim=3)
    if extra is not None and extra.shape[2] != 3:
        raise ValueError("Additional candidates must contain XYZ")
    add_v3 = bool(f.get("add_v3_experts",False))
    if add_v3 and v3 is None:
        raise ValueError("Adding V3 experts requires the V3 feature block")
    layout = dict(experts=e.shape[1], context_width=0 if context is None else context.shape[1],
                  extra=0 if extra is None else extra.shape[1], v3_width=0 if v3 is None else v3.shape[1],
                  add_v3_experts=add_v3)
    return e, v3, context, extra, layout


def _basis(a):
    lon = torch.atan2(a[:,1],a[:,0])
    east = torch.stack((-lon.sin(),lon.cos(),torch.zeros_like(lon)),dim=1)
    return east, torch.linalg.cross(a,east,dim=1)


def _local(anchor, points):
    east,north = _basis(anchor)
    plane = torch.stack(((points*east[:,None]).sum(-1),(points*north[:,None]).sum(-1)),dim=2)
    cosine = (points*anchor[:,None]).sum(-1).clamp(-1,1)
    sine = torch.linalg.vector_norm(plane,dim=-1)
    angle = torch.atan2(sine,cosine)
    tangent = plane * (angle/sine.clamp_min(1e-8)).unsqueeze(-1)
    anti = (sine < 1e-7) & (cosine < 0)
    tangent[:,:,0] = torch.where(anti,angle,tangent[:,:,0])
    tangent[:,:,1] = torch.where(anti,0.,tangent[:,:,1])
    return tangent


def _exp(a,step):
    east,north = _basis(a)
    length = torch.linalg.vector_norm(step,dim=1,keepdim=True)
    step = step * (math.pi/length.clamp_min(1e-8)).clamp(max=1)
    length = length.clamp(max=math.pi)
    tangent = step[:,:1]*east + step[:,1:]*north
    return F.normalize(a*length.cos()+tangent*torch.sinc(length/math.pi),dim=1)


def _normal(x):
    # Gauss-map normals yield geodetic latitudes, including polar directions.
    return F.normalize(torch.cat((x[...,:2],x[...,2:3]/(1-E2)),dim=-1),dim=-1)


def _angle(a,b):
    return 2*torch.atan2(torch.linalg.vector_norm(a-b,dim=-1),torch.linalg.vector_norm(a+b,dim=-1))


def _prepare(experts,f,dev):
    values,v3,ctxt,extra,layout = _inputs(experts,f)
    if not len(values):
        return None, layout
    n=len(values)
    parts=[]
    for lo in range(0,n,8192):
        sl=slice(lo,lo+8192)
        to=lambda x: torch.as_tensor(x,device=dev,dtype=torch.float32)
        raw=[to(values[sl])]
        context=[]
        if extra is not None:
            raw.append(to(extra[sl]))
        if layout["add_v3_experts"]:
            raw.append(to(v3[sl,:21]).reshape(-1,7,3))
        fixed=torch.cat(raw,dim=1)
        if (torch.linalg.vector_norm(fixed,dim=2) < .01).any():
            raise ValueError("Main expert directions must be nonzero")
        anchor=F.normalize(fixed[:,0],dim=1)
        prior=torch.ones(fixed.shape[:2],device=dev)
        if v3 is not None:
            source=to(v3[sl])
            ranked=torch.stack([source[:,k:k+40].reshape(-1,10,4)[:,:RANKS] for k in (21,74,127)],dim=1)
            prob=ranked[:,:,:,0].clamp(0,1)
            pt=ranked[:,:,:,1:].reshape(len(anchor),3*RANKS,3)
            valid=(torch.linalg.vector_norm(pt,dim=2)>.01)&(prob.flatten(1)>1e-10)
            pt=torch.where(valid[:,:,None],pt,anchor[:,None])
            raw.append(pt)
            mass=(prob/(prob.sum(2,keepdim=True).clamp_min(1e-9))).flatten(1)
            prior=torch.cat((prior*.67/fixed.shape[1],torch.where(valid,.33*mass/3,0.)),dim=1)
            context.extend([prob.flatten(1), *[source[:,k+40:k+53] for k in (21,74,127)],
                            source[:,180:225],source[:,246:]])
        if ctxt is not None:
            context.append(to(ctxt[sl]))
        e=F.normalize(torch.cat(raw,dim=1),dim=2)
        t=_local(anchor,e)
        t[:,0]=0
        radii=t.norm(dim=2)
        # Angular displacement and agreement retain precision at street scale.
        med=torch.median(t,dim=1).values
        z=torch.cat([anchor,t.flatten(1),torch.log1p(80*radii),
                     torch.log1p(80*(t-med[:,None]).norm(dim=2)),*context],dim=1)
        lp=torch.where(prior>0,prior.clamp_min(1e-12).log(),-50.)
        parts.append((e,t,z,lp))
    blocks=tuple(torch.cat([part[i] for part in parts],0) for i in range(4))
    return blocks, layout


class _Gate(nn.Module):
    def __init__(self,nin,m,recipe):
        super().__init__()
        out=m
        if recipe=="residual":out+=2
        if recipe=="cdf":out*=len(RISK_KNOTS_KM)
        if recipe=="laplace":out*=2
        if recipe=="linear":
            self.head=nn.Linear(nin,out)
        else:
            self.head=nn.Sequential(nn.Linear(nin,72),nn.LayerNorm(72),nn.SiLU(),
                                    nn.Linear(72,48),nn.SiLU(),nn.Linear(48,out))
        last=self.head if recipe=="linear" else self.head[-1]
        nn.init.normal_(last.weight,std=.008)
        nn.init.zeros_(last.bias)

    def forward(self,z):
        return self.head(z)


def _width(raw,m):
    # Inverse angular scale, approximately 12--3000 km. The spherical
    # normalizer for exp(-kappa*angle) is proportional to (1+exp(-pi*k))/(1+k²).
    return torch.exp(1.3+5*raw[:,m:].sigmoid())


def _median(t,w,width=None):
    pos=(w[:,:,None]*t).sum(1)
    smooth=.00065 if width is None else width
    for _ in range(5):
        r=t-pos[:,None]
        ratio=w*torch.rsqrt(r.square().sum(2)+smooth**2)
        pos=(ratio[:,:,None]*t).sum(1)/ratio.sum(1,keepdim=True).clamp_min(1e-12)
    return pos


def _cdf_risk(raw,m):
    pred=raw.reshape(len(raw),m,len(RISK_KNOTS_KM)).sigmoid()
    # Each term approximates the area under the survival curve. Empirically
    # reliable near-distance probabilities also contribute to the final risk.
    knots=raw.new_tensor((0.,*RISK_KNOTS_KM,20020.))/RADIUS_KM
    widths=(knots[2:]-knots[:-2])*.5
    return (pred*widths).sum(2)


def _weights(raw,t,lp,spec):
    m=t.shape[1]
    recipe=spec["recipe"]
    if recipe=="risk":
        relative=raw[:,:m].tanh()*(t.norm(dim=2)+.004)
        logits=-relative/.012+.25*lp
    elif recipe=="cdf":
        logits=-_cdf_risk(raw,m)/.020+.25*lp
    else:
        logits=raw[:,:m]+lp
    return (logits/spec["temperature"]).masked_fill(lp< -40,-60.)


def _prediction(raw,e,t,lp,spec):
    m=t.shape[1]; mode=spec["decode"]
    logits=_weights(raw,t,lp,spec)
    if mode in ("hard","mode"):
        if mode=="mode":
            k=_width(raw,m)
            dot=(e@e.transpose(1,2)).clamp(-1,1)
            distance=dot.acos()
            log_density=logits.log_softmax(1)+torch.log1p(k.square())-torch.log1p(torch.exp(-math.pi*k))
            logits=torch.logsumexp(log_density[:,None]-distance*k[:,None],dim=2)
            logits=logits.masked_fill(lp< -40,-60.)
        return e[torch.arange(len(e),device=e.device),logits.argmax(1)]
    w=logits.softmax(1)
    if mode=="mean":
        step=(w[:,:,None]*t).sum(1)
    else:
        scale=_width(raw,m).reciprocal() if mode=="density" else None
        step=_median(t,w,scale)
        if mode=="residual":
            spread=(w*(t-step[:,None]).norm(dim=2)).sum(1,keepdim=True)
            step=step+(.002+.55*spread).clamp(max=.09)*raw[:,m:m+2].tanh()
    return _exp(e[:,0],step)


def _loss(raw,e,t,lp,y,dist,recipe):
    m=t.shape[1]
    if recipe in ("linear","mean","median","residual"):
        mode={"linear":"mean","mean":"mean","median":"median","residual":"residual"}[recipe]
        p=_prediction(raw,e,t,lp,{"recipe":recipe,"decode":mode,"temperature":1.})
        return 30*_angle(_normal(p),y).mean()
    valid=(lp> -40).float()
    if recipe=="risk":
        # A distance difference cancels much of the shared outlier noise.
        scale=t.norm(dim=2)+.004
        target=((dist-dist[:,:1])/scale).clamp(-1.02,1.02)
        importance=(.025+t.norm(dim=2)).clamp(max=.45)*valid
        return ((raw.tanh()-target).square()*importance).sum()/importance.sum().clamp_min(1)
    if recipe=="oracle":
        target=(-dist/.013+.25*lp).masked_fill(lp< -40,-60.).softmax(1)
        return -(target*(raw+lp).log_softmax(1)).sum(1).mean()
    if recipe=="cdf":
        thresholds=raw.new_tensor(RISK_KNOTS_KM)/RADIUS_KM
        target=(dist[:,:,None]>thresholds).float()
        logit=raw.reshape_as(target)
        error=F.binary_cross_entropy_with_logits(logit,target,reduction="none").mean(2)
        return (valid*error).sum()/valid.sum().clamp_min(1)
    if recipe=="laplace":
        k=_width(raw,m)
        logdensity=torch.log1p(k.square())-torch.log1p(torch.exp(-math.pi*k))-k*dist
        return -torch.logsumexp((raw[:,:m]+lp).log_softmax(1)+logdensity,dim=1).mean()
    raise KeyError(recipe)


def _standardize(z,center,scale):
    return ((z-center)/scale).clamp(-7,7)


def fit_grid(experts_train,y_train,f=None,*,seed=812443,device=None,only=None):
    """Fit selected gates to honest training OOF columns; return CPU states."""
    specs=grid()
    keys=list(specs) if only is None else ([only] if isinstance(only,str) else list(dict.fromkeys(only)))
    for key in keys:
        _spec(key)
    if not keys:
        return {}
    dev=_device(device)
    with torch.inference_mode(False),torch.enable_grad():
        features,layout=_prepare(experts_train,f,dev)
        if features is None:
            raise ValueError("Training needs at least one out-of-fold row")
        e,t,z,lp=features
        a=_matrix(y_train,"training targets",len(e))
        if a.shape!=(len(e),3):
            raise ValueError("Expected training labels [rows, 3]")
        y=torch.as_tensor(a,dtype=torch.float32,device=dev)
        if (y.norm(dim=1)<1e-7).any():
            raise ValueError("Training targets must be nonzero")
        y=_normal(y)
        center=z.mean(0)
        scale=z.std(0,unbiased=False).clamp_min(.015)
        z=_standardize(z,center,scale)
        count=z.shape[0];m=e.shape[1]
        selected_recipes=list(dict.fromkeys(specs[k]["recipe"] for k in keys))
        use_risk=any(r in ("oracle","risk","cdf","laplace") for r in selected_recipes)
        dist=None
        if use_risk:
            dist=torch.cat([_angle(_normal(e[start:start+8192]),y[start:start+8192,None])
                            for start in range(0,count,8192)],dim=0)
        fitted={}
        for recipe in selected_recipes:
            recipe_seed=(int(seed)+1769*(RECIPES.index(recipe)+1))%(2**31-1)
            batches=torch.Generator(device=dev).manual_seed(recipe_seed+41)
            with torch.random.fork_rng(devices=([dev] if dev.type=="cuda" else [])):
                torch.manual_seed(recipe_seed)
                model=_Gate(z.shape[1],m,recipe).to(dev)
            opt=torch.optim.AdamW(model.parameters(),lr=.0035,weight_decay=.012)
            epochs=8 if recipe in ("risk","cdf","oracle") else 6
            for epoch in range(epochs):
                rate=.0035*(.35+.65*(1+math.cos(math.pi*epoch/max(1,epochs-1)))/2)
                for group in opt.param_groups:
                    group["lr"]=rate
                for ids in torch.randperm(count,generator=batches,device=dev).split(BATCH):
                    out=model(z[ids])
                    loss=_loss(out,e[ids],t[ids],lp[ids],y[ids],None if dist is None else dist[ids],recipe)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),4.)
                    opt.step()
            if not torch.isfinite(loss.detach()).item():
                raise FloatingPointError(f"Nonfinite geographic gate loss: {recipe}")
            fitted[recipe]={k:v.detach().cpu() for k,v in model.state_dict().items()}
            del opt, model
        center=center.detach().cpu()
        scale=scale.detach().cpu()
        return {name:dict(version=VERSION,spec=dict(specs[name]),layout=dict(layout),seed=int(seed),
                          center=center,scale=scale,candidate_count=m,weights=fitted[specs[name]["recipe"]])
                for name in keys}


def fit_one(spec,experts_train,y_train,f=None,*,seed=812443,device=None):
    """Fit a single grid specification with the same recipe seed as fit_grid."""
    name=_spec(spec)["name"]
    return fit_grid(experts_train,y_train,f,seed=seed,device=device,only=[name])[name]


@torch.inference_mode()
def predict_grid(states,experts_query,f=None,*,device=None):
    """Predict a collection of states from the same fit_grid in bounded batches."""
    if not states:
        return {}
    dev=_device(device)
    data,layout=_prepare(experts_query,f,dev)
    for name,state in states.items():
        if state.get("version")!=VERSION or state["layout"]!=layout or _spec(state["spec"])["name"]!=name:
            raise ValueError("Incompatible MAE gate feature layout or checkpoint")
    if data is None:
        return {name:np.empty((0,3),np.float32) for name in states}
    e,t,z,lp=data
    predictions={name:np.empty((len(e),3),np.float32) for name in states}
    # A recipe trained once can serve multiple actions from the same logits.
    # Independently loaded checkpoints may use different weights or scales,
    # so group only states sharing the actual tensor objects from fit_grid.
    grouped={}
    for name,state in states.items():
        if state["candidate_count"]!=e.shape[1] or len(state["center"])!=z.shape[1]:
            raise ValueError("Unexpected number of candidates or features")
        key=(id(state["weights"]),id(state["center"]),id(state["scale"]),state["spec"]["recipe"])
        grouped.setdefault(key,[]).append(name)
    for (_,_,_,recipe),names in grouped.items():
        state=states[names[0]]
        model=_Gate(z.shape[1],e.shape[1],recipe).to(dev)
        model.load_state_dict(state["weights"]);model.eval()
        center,scale=state["center"].to(dev),state["scale"].to(dev)
        for start in range(0,len(e),4096):
            sl=slice(start,start+4096)
            out=model(_standardize(z[sl],center,scale))
            for name in names:
                pred=_prediction(out,e[sl],t[sl],lp[sl],states[name]["spec"])
                predictions[name][sl]=pred.cpu().numpy()
    for name,value in predictions.items():
        if not np.isfinite(value).all() or np.max(abs(np.linalg.norm(value,axis=1)-1))>5e-5:
            raise FloatingPointError(f"Invalid geographic MAE prediction: {name}")
    return predictions


def predict(state,experts_query,f=None,*,device=None):
    name=state["spec"]["name"]
    return predict_grid({name:state},experts_query,f,device=device)[name]


def save(path,state,*,members=None):
    """Atomically write learned tensors and all rendering/feature metadata."""
    if state.get("version")!=VERSION:
        raise ValueError("Unknown checkpoint version")
    _spec(state["spec"])
    obj=dict(state)
    if members is not None:
        if len(members)!=state["layout"]["experts"]:
            raise ValueError("Member names must cover the input experts in order")
        obj["members"]=[str(name) for name in members]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix="."+path.name,suffix=".tmp",delete=False) as tmp:
        temp=Path(tmp.name)
    try:
        torch.save(obj,temp)
        os.replace(temp,path)
    finally:
        temp.unlink(missing_ok=True)


def load(path):
    state=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(state,dict) or state.get("version")!=VERSION:
        raise ValueError("Unsupported geographic MAE gate checkpoint")
    _spec(state["spec"])
    return state


def predict_saved(path,experts_query,f=None,*,device=None):
    return predict(load(path),experts_query,f,device=device)
