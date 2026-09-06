import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(c):
    g=min(32,c)
    while c%g: g-=1
    return nn.GroupNorm(g,c)

class ResBlock(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        self.n1=norm(cin); self.c1=nn.Conv2d(cin,cout,3,padding=1)
        self.n2=norm(cout); self.c2=nn.Conv2d(cout,cout,3,padding=1)
        self.skip=nn.Identity() if cin==cout else nn.Conv2d(cin,cout,1)
    def forward(self,x):
        h=self.c1(F.silu(self.n1(x)))
        h=self.c2(F.silu(self.n2(h)))
        return h+self.skip(x)

class ProjectedVQ(nn.Module):
    def __init__(self,num_codes=8192,dim=256,lookup_dim=256,beta=.25):
        super().__init__()
        self.to_lookup=nn.Conv2d(dim,lookup_dim,1)
        self.from_lookup=nn.Conv2d(lookup_dim,dim,1)
        self.codebook=nn.Embedding(num_codes,lookup_dim)
        self.beta=beta
    def forward(self,z):
        z=self.to_lookup(z)
        b,c,h,w=z.shape
        flat=z.permute(0,2,3,1).reshape(-1,c)
        fn=F.normalize(flat,dim=-1); cn=F.normalize(self.codebook.weight,dim=-1)
        ids=torch.argmax(fn@cn.t(),dim=-1)
        q=self.codebook(ids).view(b,h,w,c).permute(0,3,1,2).contiguous()
        loss=F.mse_loss(q,z.detach())+self.beta*F.mse_loss(z,q.detach())
        q=z+(q-z).detach()
        return self.from_lookup(q),ids.view(b,h,w),loss
    @torch.no_grad()
    def decode_ids(self,ids):
        q=self.codebook(ids).permute(0,3,1,2).contiguous()
        return self.from_lookup(q)

class ImageTokenizer(nn.Module):
    def __init__(self,codebook_size=8192,embed_dim=256,base_channels=64,
                 channel_mults=(1,2,4,4,8),num_res_blocks=2,
                 projection_dim=256,commitment_beta=.25):
        super().__init__()
        assert len(channel_mults)==5, '5 levels create 4 downsamples = factor 16'
        enc=[]; ch=base_channels
        enc.append(nn.Conv2d(3,ch,3,padding=1))
        for i,m in enumerate(channel_mults):
            out=base_channels*m
            for _ in range(num_res_blocks): enc += [ResBlock(ch,out)]; ch=out
            if i<len(channel_mults)-1: enc += [nn.Conv2d(ch,ch,4,2,1)]
        enc += [norm(ch),nn.SiLU(),nn.Conv2d(ch,embed_dim,1)]
        self.encoder=nn.Sequential(*enc)
        self.vq=ProjectedVQ(codebook_size,embed_dim,projection_dim,commitment_beta)

        dec=[]; rev=list(channel_mults)[::-1]; ch=base_channels*rev[0]
        dec += [nn.Conv2d(embed_dim,ch,3,padding=1)]
        for i,m in enumerate(rev):
            out=base_channels*m
            for _ in range(num_res_blocks): dec += [ResBlock(ch,out)]; ch=out
            if i<len(rev)-1:
                dec += [nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(ch,ch,3,padding=1)]
        dec += [norm(ch),nn.SiLU(),nn.Conv2d(ch,3,3,padding=1),nn.Tanh()]
        self.decoder=nn.Sequential(*dec)

    def encode(self,x):
        z=self.encoder(x)
        return self.vq(z)
    @torch.no_grad()
    def tokenize(self,x):
        _,ids,_=self.encode(x)
        return ids.flatten(1)
    def forward(self,x):
        q,ids,qloss=self.encode(x)
        return {'reconstruction':self.decoder(q),'ids':ids,'quantized':q,'quantization_loss':qloss}
