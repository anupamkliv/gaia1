import random
import torch
TASKS=('image','video','autoregressive','interpolation')
def sample_task(): return random.choice(TASKS)
def masks(task,b,t,device):
    context=torch.zeros(b,t,device=device); tokens=torch.ones(b,t,device=device)
    if task=='autoregressive': context[:,:max(1,t//2)]=1
    elif task=='interpolation': context[:,0]=1; context[:,-1]=1; tokens.zero_()
    return context,tokens

def diffusion_loss(pred,target,frame_mask=None,l1=.1,l2=1.):
    e=pred-target
    if frame_mask is not None: e=e*frame_mask[:,None,:,None,None]
    return l1*e.abs().mean()+l2*e.pow(2).mean()
