"""Ten compact regression architectures for the preliminary GPU screen."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS={
    "shallow_mlp":"Один скрытый слой, квадратичная ошибка координат.",
    "residual_angle":"Остаточные блоки и квадрат центрального угла.",
    "group_gate":"Две группы входных признаков с обучаемым покомпонентным шлюзом.",
    "expert_mixture":"Четыре независимые локальные головы и обучаемые веса.",
    "vmf_mixture":"Смесь трёх направлений фон Мизеса–Фишера.",
    "token_attention":"Самовнимание по токенам групп признаков.",
    "bilinear":"Низкоранговое мультипликативное объединение.",
    "random_kernel":"Регуляризованная регрессия случайных Fourier-признаков.",
    "robust_mlp":"Регрессия с устойчивой ошибкой Huber и дополнительным угловым штрафом.",
    "regional_heads":"Восемь географических голов с вероятностным маршрутизатором.",
}


def norm(x):
    return F.normalize(x.float(),dim=-1,eps=1e-8)


def angular(p,y):
    cross=torch.linalg.cross(p,y,dim=1).norm(dim=1)
    dot=(p*y).sum(dim=1).clamp(-1,1)
    return torch.atan2(cross.clamp_min(1e-7),dot)


class Residual(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),
                               nn.Dropout(.1),nn.Linear(width,width))
    def forward(self,x):
        return x+self.net(x)


class Model(nn.Module):
    def __init__(self,dim,method):
        super().__init__()
        self.method=method
        self.dim=dim
        self.width=256
        self.part=(dim+1)//2
        if method=="group_gate":
            self.left=nn.Sequential(nn.Linear(self.part,256),nn.GELU())
            self.right=nn.Sequential(nn.Linear(dim-self.part,256),nn.GELU())
            self.mix=nn.Linear(512,256)
            self.head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,3))
        elif method=="token_attention":
            self.token=32
            self.n_tokens=(dim+self.token-1)//self.token
            self.project=nn.Linear(self.token,64)
            layer=nn.TransformerEncoderLayer(64,4,128,dropout=.08,batch_first=True,activation="gelu")
            self.attention=nn.TransformerEncoder(layer,1)
            self.embedding=nn.Parameter(torch.zeros(1,self.n_tokens,64))
            self.head=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,3))
        elif method=="bilinear":
            self.left=nn.Linear(dim,256)
            self.right=nn.Linear(dim,256)
            self.head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,3))
        else:
            layers=[nn.Linear(dim,self.width),nn.LayerNorm(self.width),nn.GELU(),nn.Dropout(.12)]
            if method=="residual_angle":
                layers.extend([Residual(self.width),Residual(self.width)])
            elif method!="shallow_mlp":
                layers.extend([nn.Linear(self.width,self.width),nn.GELU(),nn.Dropout(.10)])
            self.backbone=nn.Sequential(*layers)
            count={"expert_mixture":4,"vmf_mixture":3,"regional_heads":8}.get(method,1)
            self.count=count
            self.dirs=nn.Linear(self.width,3*count)
            self.weights=nn.Linear(self.width,count) if count>1 else None

    def forward(self,x):
        if self.method=="group_gate":
            a,b=self.left(x[:,:self.part]),self.right(x[:,self.part:])
            g=self.mix(torch.cat([a,b],dim=1)).sigmoid()
            return norm(self.head(a*g+b*(1-g))),None
        if self.method=="token_attention":
            value=F.pad(x,(0,self.n_tokens*self.token-self.dim)).reshape(len(x),self.n_tokens,self.token)
            hidden=self.attention(self.project(value)+self.embedding)
            return norm(self.head(hidden.mean(dim=1))),None
        if self.method=="bilinear":
            return norm(self.head(torch.tanh(self.left(x))*torch.tanh(self.right(x)))),None
        hid=self.backbone(x)
        dirs=norm(self.dirs(hid).reshape(len(x),self.count,3))
        if self.weights is None:
            return dirs[:,0],None
        logits=self.weights(hid).float()
        weight=logits.softmax(dim=1)
        return norm((weight[:,:,None]*dirs).sum(dim=1)),(dirs,logits)


def prepare_xy(X_train,X_query):
    train=torch.as_tensor(X_train,device="cuda",dtype=torch.float32)
    query=torch.as_tensor(X_query,device="cuda",dtype=torch.float32)
    mean=train.mean(0)
    scale=train.std(0).clamp_min(1e-4)
    return ((train-mean)/scale).clamp(-12,12).half(),((query-mean)/scale).clamp(-12,12).half(),mean,scale


def fit_predict(method,X_train,Y_train,X_query,seed,epochs=6,save_path=None):
    torch.manual_seed(seed)
    x,q,avg,std=prepare_xy(X_train,X_query)
    y=torch.as_tensor(Y_train,device="cuda",dtype=torch.float32)
    if method=="random_kernel":
        W=torch.randn(x.shape[1],384,device="cuda")/np.sqrt(x.shape[1])
        phase=torch.rand(384,device="cuda")*2*np.pi
        phi=(x.float()@W+phase).cos()
        z=(q.float()@W+phase).cos()
        gram=phi.T@phi
        gram.diagonal().add_(.01*len(phi))
        coef=torch.linalg.solve(gram,phi.T@y)
        if save_path is not None:
            torch.save({"method":method,"mean":avg.cpu(),"std":std.cpu(),"W":W.cpu(),
                        "phase":phase.cpu(),"coef":coef.cpu()},save_path)
        return norm(z@coef).cpu().numpy()
    model=Model(x.shape[1],method).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.003,fused=True)
    scaler=torch.amp.GradScaler("cuda")
    lr=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00015)
    batch_size=1024 if method=="token_attention" else 4096
    regional=None
    if method=="regional_heads":
        regional=((y[:,0]>0).long()*4+(y[:,1]>0).long()*2+(y[:,2]>0).long())
    for _ in range(epochs):
        for idx in torch.randperm(len(x),device="cuda").split(batch_size):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                pred,extra=model(x[idx])
            truth=y[idx]
            angle=angular(pred,truth)
            if method=="residual_angle":
                loss=angle.square().mean()
            elif method=="robust_mlp":
                loss=F.smooth_l1_loss(pred,truth,beta=.15)+.12*angle.square().mean()
            elif method=="vmf_mixture":
                dirs,logits=extra
                concentration=torch.tensor([4.,9.,18.],device="cuda")
                ll=concentration*((dirs*truth[:,None]).sum(dim=2)-1)+concentration.log()
                loss=-torch.logsumexp(logits.log_softmax(dim=1)+ll,dim=1).mean()+.3*angle.square().mean()
            elif method=="regional_heads":
                dirs,logits=extra
                chosen=dirs[torch.arange(len(idx),device="cuda"),regional[idx]]
                loss=F.mse_loss(chosen,truth)+.35*F.cross_entropy(logits,regional[idx])+.25*angle.square().mean()
            else:
                loss=F.mse_loss(pred,truth)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        lr.step()
    if save_path is not None:
        torch.save({"method":method,"dim":x.shape[1],"mean":avg.cpu(),"std":std.cpu(),
                    "state":{k:v.cpu() for k,v in model.state_dict().items()}},save_path)
    model.eval()
    answer=[]
    with torch.inference_mode(),torch.autocast("cuda"):
        for block in q.split(batch_size):
            answer.append(model(block)[0].cpu().numpy())
    return np.concatenate(answer)
