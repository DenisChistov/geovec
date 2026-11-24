"""One hundred ensembles: ten calibration-only expert groups by ten fusion rules."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

from meta_models import logmap,expmap,unit

GROUPS={
    "best_rmse":"Восемь лучших по RMSE.",
    "best_mae":"Восемь лучших по MAE.",
    "balanced_bases":"Лучшие из нейросетей, геоклассификаторов и поиска.",
    "neural":"Лучшие прямые нейрорегрессоры.",
    "geo":"Лучшие вероятностные геомодели.",
    "retrieval":"Лучшие модели поиска.",
    "feature_diverse":"Лучшие для различных структур признаков.",
    "decorrelated":"Сильные модели со слабо согласованными ошибками.",
    "tail_safe":"Лучшие по 95-му перцентилю ошибок.",
    "near_precision":"Лучшие по доле малых ошибок.",
}
METHODS={
    "uniform":"Равные веса и сферическое среднее.",
    "inverse_risk":"Веса, обратные квадрату RMSE.",
    "soft_risk":"Экспоненциальные веса калибровочного риска.",
    "median":"Геометрическая медиана на сфере.",
    "medoid":"Взвешенный кандидат с наименьшим суммарным квадратом расстояния до экспертов.",
    "signed_wgs84":"Регуляризованная регрессия смещений восток/север.",
    "simplex":"Выпуклая комбинация экспертов, оптимизированная по квадратичной ошибке.",
    "regional_risk":"Локальный риск экспертов по похожим географическим калибровочным прогнозам.",
    "context_gate":"Ridge-прогноз риска по географии и расхождению экспертных ответов.",
    "trim_consensus":"Подавление отличающихся от согласованной группы экспертов.",
}


def angular(a,b):
    return np.arccos(np.clip((a*b).sum(-1),-1,1))


def risk_matrix(pred,y):
    # Angular error provides a fast, continuous training-only proxy for grouping.
    return angular(pred,y[:,None,:])*6371.0088


def mean(points,weights,median=False):
    weights=np.broadcast_to(np.asarray(weights),points.shape[:2])
    p=unit((weights[:,:,None]*points).sum(1))
    for _ in range(3):
        dot=(points*p[:,None,:]).sum(2).clip(-1,1)
        tangent=points-dot[:,:,None]*p[:,None,:]
        norm=np.linalg.norm(tangent,axis=2)
        angle=np.arctan2(norm,dot)
        local=weights*(angle/np.maximum(norm,1e-7))
        if median:
            local/=np.maximum(angle,1e-3)
            local/=np.maximum((weights/np.maximum(angle,1e-3)).sum(1,keepdims=True),1e-7)
        step=(local[:,:,None]*tangent).sum(1)
        length=np.linalg.norm(step,axis=1,keepdims=True)
        p=unit(np.cos(length)*p+np.sinc(length/np.pi)*step)
    return p


def select_groups(names,cal,truth):
    distance=risk_matrix(cal,truth)
    mse=np.mean(distance**2,axis=0)
    rank=np.argsort(mse)
    families=np.array([name.split("/")[0] for name in names])
    features=np.array([name.split("/")[1] for name in names])
    out={}
    out["best_rmse"]=rank[:8]
    out["best_mae"]=np.argsort(distance.mean(0))[:8]
    per_family=[rank[families[rank]==s] for s in ("neural","geo","retrieval")]
    out["balanced_bases"]=np.concatenate([x[:3] for x in per_family])
    for key,values in zip(("neural","geo","retrieval"),per_family):
        out[key]=values[:8]
    out["feature_diverse"]=np.array([rank[features[rank]==f][0] for f in sorted(set(features)) if f!="prior"])[:10]
    chosen=[int(rank[0])]
    for _ in range(7):
        residual=distance[:,rank[:40]]
        reference=distance[:,chosen].mean(axis=1)
        centered=residual-residual.mean(0)
        r=reference-reference.mean()
        corr=(centered*r[:,None]).sum(0)/(np.sqrt((centered**2).sum(0)*(r*r).sum())+1e-7)
        objective=mse[rank[:40]]/mse[rank[0]]+.35*corr
        objective[np.isin(rank[:40],chosen)]=np.inf
        chosen.append(int(rank[np.argmin(objective)]))
    out["decorrelated"]=np.array(chosen)
    out["tail_safe"]=np.argsort(np.quantile(distance,.95,axis=0))[:8]
    out["near_precision"]=np.argsort(-(distance<100).mean(0))[:8]
    return out


def _features(e):
    p=mean(e,np.ones(e.shape[1])/e.shape[1])
    gap=angular(e,p[:,None])
    return p,np.column_stack([p,gap,gap.mean(1),gap.std(1),gap.max(1)])


def _softmax(z):
    z=z-z.max(1,keepdims=True)
    e=np.exp(z)
    return e/e.sum(1,keepdims=True)


def one(method,train,truth,query):
    m=train.shape[1]
    distance=risk_matrix(train,truth)
    mse=np.mean(distance**2,axis=0)
    global_weight=(1/(mse+1e-8));global_weight/=global_weight.sum()
    if method=="uniform":
        return mean(query,np.ones(m)/m)
    if method=="inverse_risk":
        return mean(query,global_weight)
    if method=="soft_risk":
        weights=np.exp(-(mse-mse.min())/max(mse.mean()*.15,1))
        return mean(query,weights/weights.sum())
    if method=="median":
        return mean(query,global_weight,median=True)
    if method=="medoid":
        dot=np.einsum("nmd,nkd->nmk",query,query).clip(-1,1)
        cost=(np.arccos(dot)**2*global_weight[None,None,:]).sum(2)
        return query[np.arange(len(query)),cost.argmin(1)]
    if method=="trim_consensus":
        p=mean(query,global_weight)
        gap=angular(query,p[:,None])
        weights=np.exp(-gap/np.maximum(np.median(gap,axis=1,keepdims=True),.02))*global_weight
        return mean(query,weights/weights.sum(1,keepdims=True))
    anchor=mean(train,global_weight)
    q_anchor=mean(query,global_weight)
    if method=="signed_wgs84":
        x=np.stack([logmap(anchor,train[:,i]) for i in range(m)],axis=2).reshape(-1,m)
        y=logmap(anchor,truth).reshape(-1)
        coef=np.linalg.solve(x.T@x+np.eye(m)*len(x)*.002,x.T@y)
        q=np.stack([logmap(q_anchor,query[:,i]) for i in range(m)],axis=2)
        return expmap(q_anchor,(q@coef).astype(np.float32))
    if method=="simplex":
        a=train.transpose(0,2,1).reshape(-1,m)
        y=truth.reshape(-1)
        gram=a.T@a/len(a); cross=a.T@y/len(a)
        objective=lambda w:float(w@gram@w-2*cross@w)
        weights=minimize(objective,global_weight.astype(float),bounds=[(0,1)]*m,
                         constraints={"type":"eq","fun":lambda w:w.sum()-1},method="SLSQP").x
        return mean(query,weights/weights.sum())
    if method=="regional_risk":
        ids=cKDTree(anchor).query(q_anchor,k=min(128,len(anchor)),workers=2)[1]
        local=np.mean(distance[ids]**2,axis=1)
        weight=1/(.7*local+.3*mse+1e-8)
        return mean(query,weight/weight.sum(1,keepdims=True))
    if method=="context_gate":
        _,X=_features(train)
        _,Z=_features(query)
        avg=X.mean(0); scale=X.std(0).clip(.01)
        X=(X-avg)/scale;Z=(Z-avg)/scale
        X=np.column_stack([X,np.ones(len(X))]);Z=np.column_stack([Z,np.ones(len(Z))])
        coef=np.linalg.solve(X.T@X+np.eye(X.shape[1])*len(X)*.02,X.T@np.log1p(distance/250))
        return mean(query,_softmax(-(Z@coef)*2))
    raise ValueError(method)
