"""Ten model families and ten feature views for cross-fitted meta-regression."""
import numpy as np
from pyproj import Geod
import torch
from torch import nn
from torch.nn import functional as F

METHODS={
    "ridge_xyz":"Глобальная ridge-регрессия трёх координат.",
    "ridge_residual":"Ridge на остатке относительно ансамблевого направления.",
    "ridge_wgs84":"Линейная корректировка восток/север на WGS84.",
    "ridge_interaction":"Нелинейный признак взаимодействия с направлением ансамбля.",
    "cat_xyz":"CatBoost MultiRMSE, прямое предсказание координат.",
    "cat_wgs84":"CatBoost MultiRMSE на геодезических остатках.",
    "mlp_xyz":"Нейросетевой регрессор трёх координат.",
    "mlp_gate":"Нейросетевой шлюз для семи экспертов.",
    "mlp_wgs84":"Нейросетевой регрессор геодезических остатков.",
    "kernel_ridge":"Случайные Fourier features и ridge.",
}
VIEWS={
    "all":"Все сохранённые прогнозы, вероятности, диагностические признаки и PCA.",
    "experts":"Координаты семи экспертов и их попарное согласие.",
    "posterior":"Три блока распределений по географическим ячейкам.",
    "retrieval":"Статистика поиска и оценки локальной плотности.",
    "uncertainty":"Энтропии, массы и географический разброс.",
    "geometry":"Координаты экспертов, их взаимные отклонения и уверенность.",
    "visual_pca":"32 компоненты PCA исходных эмбеддингов.",
    "interactions":"Произведения координат экспертов на признаки уверенности.",
    "rank_shape":"Ранги вероятностей и пространственное рассеяние центров.",
    "compact":"Ключевые вероятностные и географические признаки вместе с PCA.",
}
GEOD=Geod(ellps="WGS84")
E2=1-(1-1/298.257223563)**2


def views(f,e):
    r=[f[:,i:i+53] for i in (21,74,127)]
    u=np.column_stack([*[a[:,40:53] for a in r],f[:,180:198]])
    coords=e.reshape(len(e),-1)
    spread=np.column_stack([(e[:,i]*e[:,j]).sum(1) for i in range(7) for j in range(i+1,7)])
    principal=f[:,246:]
    yield "all",f
    yield "experts",np.column_stack([coords,spread])
    yield "posterior",f[:,21:180]
    yield "retrieval",f[:,180:225]
    yield "uncertainty",u
    yield "geometry",np.column_stack([coords,spread,u])
    yield "visual_pca",principal
    yield "interactions",(coords[:,:,None]*u[:,None,:8]).reshape(len(e),-1)
    yield "rank_shape",np.column_stack([*[a[:,:40].reshape(-1,10,4)[:,:,0] for a in r],spread])
    yield "compact",np.column_stack([coords,u,principal])


def xyz_lonlat(v):
    x,y,z=np.asarray(v,dtype=np.float64).T
    return np.degrees(np.arctan2(y,x)),np.degrees(np.arctan2(z,(1-E2)*np.hypot(x,y)))


def logmap(anchor,truth):
    lon,lat=xyz_lonlat(anchor)
    lo,la=xyz_lonlat(truth)
    az,_,distance=GEOD.inv(lon,lat,lo,la)
    rad=np.radians(az)
    return np.column_stack([np.sin(rad)*distance,np.cos(rad)*distance]).astype(np.float32)/6371008.8


def expmap(anchor,offset):
    lon,lat=xyz_lonlat(anchor)
    direction=np.degrees(np.arctan2(offset[:,0],offset[:,1]))
    d=np.linalg.norm(offset,axis=1).clip(0,np.pi)*6371008.8
    lon,lat,_=GEOD.fwd(lon,lat,direction,d)
    lon,lat=np.radians(lon),np.radians(lat)
    return unit(np.column_stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),(1-E2)*np.sin(lat)]))


def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-8)


def ridge(x,y,z,penalty=.016):
    ones=torch.ones(len(x),1,device=x.device)
    x=torch.cat([x,ones],dim=1)
    z=torch.cat([z,torch.ones(len(z),1,device=z.device)],dim=1)
    gram=x.T@x
    gram.diagonal().add_(penalty*len(x))
    coef=torch.linalg.solve(gram,x.T@y)
    return z@coef,coef


def fit_predict(method,features_train,Y,features_query,experts_train,experts_query,seed,save_path=None):
    torch.manual_seed(seed)
    x=torch.as_tensor(features_train,device="cuda")
    z=torch.as_tensor(features_query,device="cuda")
    mean=x.mean(0)
    sd=x.std(0).clamp_min(1e-5)
    x=((x-mean)/sd).clamp(-12,12)
    z=((z-mean)/sd).clamp(-12,12)
    anchor=experts_train[:,1]
    query_anchor=experts_query[:,1]
    target=np.asarray(Y,dtype=np.float32)
    tangent="wgs84" in method
    if tangent:
        target=logmap(anchor,Y)
    elif method in ("ridge_residual","ridge_interaction"):
        target=target-anchor
    y=torch.as_tensor(target,device="cuda")
    state={"method":method,"mean":mean.cpu(),"sd":sd.cpu()}
    if method.startswith("cat_"):
        from catboost import CatBoostRegressor
        torch.cuda.empty_cache()
        cat=CatBoostRegressor(loss_function="MultiRMSE",iterations=180,depth=5,
                             learning_rate=.08,task_type="GPU",verbose=False,
                             random_seed=seed,allow_writing_files=False,thread_count=4,
                             gpu_ram_part=.5)
        cat.fit(x.cpu().numpy(),target)
        pred=cat.predict(z.cpu().numpy()).astype(np.float32)
        if save_path is not None:
            cat.save_model(str(save_path)+".cbm")
    elif method.startswith("mlp_"):
        width=128
        out_dim=7 if method=="mlp_gate" else y.shape[1]
        model=nn.Sequential(nn.Linear(x.shape[1],width),nn.LayerNorm(width),nn.GELU(),
                            nn.Dropout(.12),nn.Linear(width,64),nn.GELU(),nn.Linear(64,out_dim)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.004,fused=True)
        features=x.half()
        e=torch.as_tensor(experts_train,device="cuda")
        eq=torch.as_tensor(experts_query,device="cuda")
        scaler=torch.amp.GradScaler("cuda")
        for _ in range(6):
            for ids in torch.randperm(len(x),device="cuda").split(2048):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    value=model(features[ids]).float()
                if method=="mlp_gate":
                    value=F.normalize((value.softmax(1)[:,:,None]*e[ids]).sum(1),dim=1)
                loss=F.mse_loss(value,y[ids])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        model.eval()
        with torch.inference_mode(),torch.autocast("cuda"):
            val=model(z.half()).float()
            if method=="mlp_gate":
                val=F.normalize((val.softmax(1)[:,:,None]*eq).sum(1),dim=1)
            pred=val.cpu().numpy()
        state["state"]={k:v.cpu() for k,v in model.state_dict().items()}
    else:
        if method=="ridge_interaction":
            a=torch.as_tensor(anchor,device="cuda")
            aq=torch.as_tensor(query_anchor,device="cuda")
            x=torch.cat([x,(x[:,:,None]*a[:,None]).flatten(1)],dim=1)
            z=torch.cat([z,(z[:,:,None]*aq[:,None]).flatten(1)],dim=1)
        elif method=="kernel_ridge":
            w=torch.randn(x.shape[1],192,device="cuda")/np.sqrt(x.shape[1])
            phase=torch.rand(192,device="cuda")*2*np.pi
            x=(x@w+phase).cos()
            z=(z@w+phase).cos()
            state.update(W=w.cpu(),phase=phase.cpu())
        output,coef=ridge(x,y,z)
        pred=output.cpu().numpy()
        state["coefficients"]=coef.cpu()
    if save_path is not None:
        torch.save(state,save_path)
    if tangent:
        return expmap(query_anchor,pred)
    if method in ("ridge_residual","ridge_interaction"):
        pred=pred+query_anchor
    if method=="mlp_xyz":
        zero=np.linalg.norm(pred,axis=1)<1e-8
        pred[zero]=query_anchor[zero]
    return unit(pred)
