import torch
@torch.no_grad()
def ddim_sample(model,schedule,image_ids,shape,device,steps=50,token_mask=None):
    x=torch.randn(shape,device=device); times=torch.linspace(schedule.steps-1,0,steps,device=device).long()
    for i,ts in enumerate(times):
        t=torch.full((shape[0],),int(ts),device=device,dtype=torch.long); v=model(x,t,image_ids,token_mask)
        x0=schedule.x0_from_v(x,v,t).clamp(-1,1); eps=schedule.eps_from_v(x,v,t)
        if i==len(times)-1: return x0
        an=schedule.abar[times[i+1]]; x=an.sqrt()*x0+(1-an).sqrt()*eps
    return x
