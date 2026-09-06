import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def gn(c):
    g=min(32,c)
    while c%g: g-=1
    return nn.GroupNorm(g,c)

class TimeEmbedding(nn.Module):
    def __init__(self,d):
        super().__init__(); self.d=d; self.mlp=nn.Sequential(nn.Linear(d,d*4),nn.SiLU(),nn.Linear(d*4,d))
    def forward(self,t):
        h=self.d//2; f=torch.exp(-math.log(10000)*torch.arange(h,device=t.device)/max(h-1,1))
        x=t.float()[:,None]*f[None]; return self.mlp(torch.cat([x.sin(),x.cos()],-1))

class Res3D(nn.Module):
    def __init__(self,cin,cout,td):
        super().__init__(); self.n1=gn(cin); self.c1=nn.Conv3d(cin,cout,3,padding=1); self.te=nn.Linear(td,cout)
        self.n2=gn(cout); self.c2=nn.Conv3d(cout,cout,3,padding=1); self.skip=nn.Identity() if cin==cout else nn.Conv3d(cin,cout,1)
    def forward(self,x,t):
        h=self.c1(F.silu(self.n1(x)))+self.te(F.silu(t))[:,:,None,None,None]
        return self.c2(F.silu(self.n2(h)))+self.skip(x)

class SpatialAttn(nn.Module):
    def __init__(self,c,h): super().__init__(); self.n=gn(c); self.a=nn.MultiheadAttention(c,h,batch_first=True)
    def forward(self,x):
        b,c,t,h,w=x.shape; y=rearrange(self.n(x),'b c t h w -> (b t) (h w) c'); y,_=self.a(y,y,y,need_weights=False)
        return x+rearrange(y,'(b t) (h w) c -> b c t h w',b=b,t=t,h=h,w=w)

class TemporalAttn(nn.Module):
    def __init__(self,c,h): super().__init__(); self.n=gn(c); self.a=nn.MultiheadAttention(c,h,batch_first=True)
    def forward(self,x):
        b,c,t,h,w=x.shape; y=rearrange(self.n(x),'b c t h w -> (b h w) t c'); y,_=self.a(y,y,y,need_weights=False)
        return x+rearrange(y,'(b h w) t c -> b c t h w',b=b,h=h,w=w)

class VideoDiffusionDecoder(nn.Module):
    '''Scalable 3D U-Net reconstruction with factorized spatial/temporal attention.'''
    def __init__(self,codebook_size=8192,token_embed_dim=256,base_channels=64,num_heads=8):
        super().__init__(); b=base_channels; td=b*4
        self.temb=TimeEmbedding(td); self.token=nn.Embedding(codebook_size,token_embed_dim); self.token_proj=nn.Conv2d(token_embed_dim,b,1)
        self.inp=nn.Conv3d(3+b,b,3,padding=1)
        self.r1=Res3D(b,b,td); self.s1=SpatialAttn(b,num_heads); self.t1=TemporalAttn(b,num_heads)
        self.down=nn.Conv3d(b,b*2,(1,4,4),stride=(1,2,2),padding=(0,1,1))
        self.r2=Res3D(b*2,b*2,td); self.s2=SpatialAttn(b*2,num_heads); self.t2=TemporalAttn(b*2,num_heads)
        self.mid=Res3D(b*2,b*2,td)
        self.upconv=nn.Conv3d(b*2,b,3,padding=1); self.r3=Res3D(b+b,b,td); self.s3=SpatialAttn(b,num_heads); self.t3=TemporalAttn(b,num_heads)
        self.out=nn.Sequential(gn(b),nn.SiLU(),nn.Conv3d(b,3,3,padding=1))

    def token_condition(self,ids,hw):
        if ids.ndim==3: ids=ids.view(ids.shape[0],ids.shape[1],18,32)
        b,t,h,w=ids.shape; e=self.token(ids).permute(0,1,4,2,3).reshape(b*t,-1,h,w); e=self.token_proj(e)
        e=F.interpolate(e,size=hw,mode='bilinear',align_corners=False)
        return e.view(b,t,-1,*hw).permute(0,2,1,3,4)

    def forward(self,x,time,image_ids,token_mask=None,temporal_enabled=True):
        c=self.token_condition(image_ids,x.shape[-2:])
        if token_mask is not None: c=c*token_mask[:,None,:,None,None]
        te=self.temb(time); h=self.inp(torch.cat([x,c],1)); h=self.r1(h,te); h=self.s1(h); h=self.t1(h) if temporal_enabled else h; skip=h
        h=self.down(h); h=self.r2(h,te); h=self.s2(h); h=self.t2(h) if temporal_enabled else h; h=self.mid(h,te)
        h=F.interpolate(h,scale_factor=(1,2,2),mode='nearest'); h=self.upconv(h)
        if h.shape[-2:]!=skip.shape[-2:]: h=F.interpolate(h,size=(h.shape[2],*skip.shape[-2:]),mode='nearest')
        h=self.r3(torch.cat([h,skip],1),te); h=self.s3(h); h=self.t3(h) if temporal_enabled else h
        return self.out(h)
