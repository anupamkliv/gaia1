import torch
import torch.nn.functional as F

def topk(logits,k=50,temp=1.):
    logits=logits/max(temp,1e-6); v,i=torch.topk(logits,min(k,logits.shape[-1]),dim=-1)
    p=F.softmax(v,dim=-1); j=torch.multinomial(p,1); return i.gather(-1,j).squeeze(-1)

@torch.no_grad()
def generate_next_frame(model,ids,speed,curvature,texts=None,k=50,temp=1.,mode='both'):
    ids=ids.clone(); current=ids.shape[1]-1; start=current*model.step+model.mt
    for i in range(model.n):
        logits=model(ids,speed,curvature,texts,mode)
        ids[:,current,i]=topk(logits[:,start+i-1],k,temp)
    return ids[:,current]

@torch.no_grad()
def cfg_logits(model,ids,speed,curvature,texts,pred_pos,conditioned='text',scale=1.):
    c=model(ids,speed,curvature,texts,conditioned)[:,pred_pos]
    u=model(ids,speed,curvature,texts,'unconditional')[:,pred_pos]
    return (1+scale)*c-scale*u
