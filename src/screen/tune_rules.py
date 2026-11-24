"""Fine-grained WGS84 ensemble hyperparameter search (development OOF only)."""
from itertools import product
from collections import defaultdict
import numpy as np
from pyproj import Geod

GEOD=Geod(ellps='WGS84')
R=6371008.8
E2=1-(1-1/298.257223563)**2


def grid():
    specs={}
    def add(pool,mode,context,alpha,tail=0.,cap=20000.):
        name=f'tuned:{pool}/{mode}/{context}/a{alpha:g}/tail{tail:g}/cap{cap:g}'
        specs[name]=dict(pool=pool,mode=mode,context=context,alpha=alpha,tail=tail,cap=cap,kind='ridge')
    for pool,mode,context,alpha in product(('core','all'),('shared','separate'),
        ('constant','spread','geo','geo_spread','expert','geo_expert'),(.0003,.001,.003,.01,.03,.1,.3,1.)):
        add(pool,mode,context,alpha)
    for mode,ctx,alpha,tail in product(('shared','separate'),('spread','geo_expert'),(.003,.03,.3),(.5,1.,2.)):
        add('all',mode,ctx,alpha,tail=tail)
    for mode,alpha,cap in product(('shared','separate'),(.003,.03,.3),(2000.,4000.,8000.)):
        add('all',mode,'geo_expert',alpha,cap=cap)
    for paper in np.arange(.05,1.,.05):
        name=f'tuned:fixed/paper{paper:.2f}'
        specs[name]={'kind':'fixed','paper':float(paper),'meta':0.}
    for meta,paper in product((.10,.20,.30,.40,.50),(.10,.20,.30,.40)):
        if meta+paper<.95:
            name=f'tuned:fixed/meta{meta:.2f}/paper{paper:.2f}'
            specs[name]={'kind':'fixed','meta':meta,'paper':paper}
    return specs


def unit(x):
    x=np.asarray(x,dtype=np.float32)
    n=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(n).all() or (n<1e-8).any():
        raise ValueError('Nonfinite or zero directions')
    return x/n


def lonlat(x):
    x,y,z=np.moveaxis(unit(x).astype(np.float64),-1,0)
    return np.degrees(np.arctan2(y,x)),np.degrees(np.arctan2(z,(1-E2)*np.hypot(x,y)))


def logmap(anchor,other):
    lon,lat=lonlat(anchor)
    olon,olat=lonlat(other)
    az,_,d=GEOD.inv(lon,lat,olon,olat)
    a=np.radians(az)
    return (np.column_stack([np.sin(a),np.cos(a)])*(d/R)[:,None]).astype(np.float32)


def geometry(experts,truth=None):
    e=unit(experts)
    anchor=e[:,0,:]
    d=np.stack([logmap(anchor,e[:,j,:]) for j in range(1,e.shape[1])],axis=2)
    a=anchor
    r=np.linalg.norm(d,axis=1)
    # Use only the query's predicted coordinates and expert disagreements.
    # No label-derived context variables are used at inference.
    spread=np.column_stack([np.log1p(12*r.max(1)),np.log1p(12*r.mean(1)),np.log1p(12*r.std(1)),
                            np.log1p(12*np.linalg.norm(d[:,:,0]-d[:,:,1],axis=1))])
    x,y,z=a.T
    geo=np.column_stack([x,y,z,x*y,x*z,y*z,x*x-y*y,z*z])
    by_expert=np.log1p(12*r)
    out={'anchor':a,'d':d,'spread':spread,'geo':geo,'expert':by_expert}
    if truth is not None:out['target']=logmap(anchor,truth)
    return out


def context(g,kind,center=None,scale=None,pool='all'):
    n=len(g['anchor'])
    if kind=='constant':
        return np.ones((n,1),np.float32),np.empty(0,np.float32),np.empty(0,np.float32)
    if pool=='core':
        d=g['d'][:,:,:3];r=np.linalg.norm(d,axis=1)
        spread=np.column_stack([np.log1p(12*r.max(1)),np.log1p(12*r.mean(1)),np.log1p(12*r.std(1)),
                                np.log1p(12*np.linalg.norm(d[:,:,0]-d[:,:,1],axis=1))])
        by_expert=np.log1p(12*r)
    else:
        spread,by_expert=g['spread'],g['expert']
    raw={'spread':lambda:spread, 'geo':lambda:g['geo'],
         'geo_spread':lambda:np.column_stack([g['geo'],spread]),
         'expert':lambda:np.column_stack([spread,by_expert]),
         'geo_expert':lambda:np.column_stack([g['geo'],spread,by_expert])}[kind]().astype(np.float32)
    if center is None:
        center=raw.mean(0);scale=raw.std(0).clip(.025)
    norm=np.clip((raw-center)/scale,-4.,4.)
    return np.column_stack([np.ones(n,np.float32),norm]).astype(np.float32),center,scale


def design(g,c,pool):
    offset=g['d'][:,:,:3] if pool=='core' else g['d']
    return (offset[:,:,:,None]*c[:,None,None,:]).reshape(len(c),2,-1).astype(np.float32)


def sufficient(g,c,pool,tail):
    x=design(g,c,pool)
    target=g['target']
    if tail:
        # Weight large incumbent errors more heavily; still choose only by
        # unweighted geodesic MSE on the held-out outer folds.
        w=np.sqrt(1+tail*np.minimum((np.linalg.norm(target,axis=1)/.30)**2,9)).astype(np.float32)
        x*=w[:,None,None]
        target=target*w[:,None]
    stats=[]
    for xx,yy in [(x.reshape(-1,x.shape[-1]),target.ravel()),(x[:,0],target[:,0]),(x[:,1],target[:,1])]:
        stats.append((np.asarray(xx.T@xx,np.float64),np.asarray(xx.T@yy,np.float64)))
    return stats


def coefficients(stats,spec,n_context):
    def solve(pair):
        gram,moment=pair
        n=len(moment)
        strength=max(np.trace(gram)/n,1e-8)*spec['alpha']
        prior=np.zeros(n)
        # Shrink toward a conservative V4-dominated ensemble. Ordering is
        # V4, meta, paper, retrieval, complementary specialists.
        prior[0]=.20;prior[n_context]=.25
        if 2*n_context<n:prior[2*n_context]=.03
        out=np.linalg.solve(gram+strength*np.eye(n),moment+strength*prior)
        return out.astype(np.float32)
    return np.stack([solve(stats[0])]*2) if spec['mode']=='shared' else np.stack([solve(stats[1]),solve(stats[2])])


def predict_geometry(g,state,x=None):
    if state['kind']=='fixed':
        step=g['d'][:,:,0]*state['meta']+g['d'][:,:,1]*state['paper']
        cap=20000.
    else:
        if x is None:
            c,_,_=context(g,state['context'],state['center'],state['scale'],state['pool'])
            x=design(g,c,state['pool'])
        step=np.einsum('ncp,cp->nc',x,state['coef'])
        cap=state['cap']
    distance=np.linalg.norm(step,axis=1)*R
    lon,lat=lonlat(g['anchor'])
    az=np.degrees(np.arctan2(step[:,0],step[:,1]))
    olon,olat,_=GEOD.fwd(lon,lat,az,np.minimum(distance,cap*1000))
    lo,la=np.radians(olon),np.radians(olat)
    return unit(np.column_stack([np.cos(la)*np.cos(lo),np.cos(la)*np.sin(lo),(1-E2)*np.sin(la)]))


def fit_one(spec,train):
    if spec['kind']=='fixed':return dict(spec)
    c,center,scale=context(train,spec['context'],pool=spec['pool'])
    stats=sufficient(train,c,spec['pool'],spec['tail'])
    return {**spec,'center':center,'scale':scale,'coef':coefficients(stats,spec,c.shape[1])}


def grouped_predictions(train,query):
    """Evaluate all grid entries without recomputing each design and Gram matrix."""
    specs=grid();groups=defaultdict(list)
    for name,spec in specs.items():
        if spec['kind']=='fixed':
            yield name,predict_geometry(query,spec)
        else:
            groups[spec['pool'],spec['context'],spec['tail']].append((name,spec))
    for (pool,kind,tail),items in groups.items():
        c,center,scale=context(train,kind,pool=pool)
        cq,_,_=context(query,kind,center,scale,pool)
        stats=sufficient(train,c,pool,tail)
        x=design(query,cq,pool)
        for name,spec in items:
            state={**spec,'coef':coefficients(stats,spec,c.shape[1])}
            yield name,predict_geometry(query,state,x)
