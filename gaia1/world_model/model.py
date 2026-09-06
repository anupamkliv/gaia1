import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel,AutoTokenizer

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6): super().__init__(); self.w=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.w

class Block(nn.Module):
    def __init__(self,d,h,ff=4,drop=0.):
        super().__init__(); self.h=h; self.hd=d//h; self.drop=drop
        self.n1=RMSNorm(d); self.qkv=nn.Linear(d,3*d,bias=False); self.o=nn.Linear(d,d,bias=False)
        self.n2=RMSNorm(d); hidden=int(d*ff*2/3)
        self.w1=nn.Linear(d,hidden,bias=False); self.w2=nn.Linear(d,hidden,bias=False); self.w3=nn.Linear(hidden,d,bias=False)
    def forward(self,x):
        b,s,d=x.shape; qkv=self.qkv(self.n1(x)).view(b,s,3,self.h,self.hd)
        q,k,v=qkv.unbind(2); q=q.transpose(1,2); k=k.transpose(1,2); v=v.transpose(1,2)
        a=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.drop if self.training else 0.)
        x=x+self.o(a.transpose(1,2).contiguous().view(b,s,d))
        h=self.n2(x); x=x+self.w3(F.silu(self.w1(h))*self.w2(h)); return x

class GaiaWorldModel(nn.Module):
    def __init__(self,codebook_size=8192,hidden_dim=512,num_layers=8,num_heads=8,
                 ffn_mult=4,max_timesteps=26,text_tokens=32,image_tokens=576,
                 action_tokens=2,t5_name='google-t5/t5-large',freeze_t5=True,dropout=0.):
        super().__init__(); self.K=codebook_size; self.d=hidden_dim; self.T=max_timesteps
        self.mt=text_tokens; self.n=image_tokens; self.la=action_tokens; self.step=self.mt+self.n+self.la
        self.image_emb=nn.Embedding(codebook_size,hidden_dim)
        self.tok=AutoTokenizer.from_pretrained(t5_name); self.t5=T5EncoderModel.from_pretrained(t5_name)
        if freeze_t5:
            for p in self.t5.parameters(): p.requires_grad=False
        self.text_proj=nn.Linear(self.t5.config.d_model,hidden_dim)
        self.speed=nn.Linear(1,hidden_dim); self.curv=nn.Linear(1,hidden_dim)
        self.null_text=nn.Parameter(torch.zeros(1,1,self.mt,hidden_dim)); self.null_action=nn.Parameter(torch.zeros(1,1,2,hidden_dim))
        self.temporal=nn.Parameter(torch.randn(max_timesteps,hidden_dim)*.02)
        self.spatial=nn.Parameter(torch.randn(self.step,hidden_dim)*.02)
        self.blocks=nn.ModuleList([Block(hidden_dim,num_heads,ffn_mult,dropout) for _ in range(num_layers)])
        self.norm=RMSNorm(hidden_dim); self.head=nn.Linear(hidden_dim,codebook_size,bias=False)

    def text_features(self,texts,b,t,device):
        if texts is None: return self.null_text.expand(b,t,-1,-1)
        flat=[texts[i][j] for i in range(b) for j in range(t)]
        z=self.tok(flat,padding='max_length',truncation=True,max_length=self.mt,return_tensors='pt').to(device)
        e=self.t5(input_ids=z.input_ids,attention_mask=z.attention_mask).last_hidden_state
        return self.text_proj(e).view(b,t,self.mt,self.d)

    def embeddings(self,image_ids,speed,curvature,texts,mode='both'):
        b,t,n=image_ids.shape; assert n==self.n and t<=self.T
        te=self.text_features(texts,b,t,image_ids.device)
        ae=torch.stack([self.speed(speed[...,None]),self.curv(curvature[...,None])],dim=2)
        if mode=='unconditional': te=self.null_text.expand(b,t,-1,-1); ae=self.null_action.expand(b,t,-1,-1)
        elif mode=='action': te=self.null_text.expand(b,t,-1,-1)
        elif mode=='text': ae=self.null_action.expand(b,t,-1,-1)
        ie=self.image_emb(image_ids)
        x=torch.cat([te,ie,ae],dim=2)
        x=x+self.temporal[:t][None,:,None,:]+self.spatial[None,None,:,:]
        return x.flatten(1,2)

    def forward(self,image_ids,speed,curvature,texts=None,mode='both'):
        x=self.embeddings(image_ids,speed,curvature,texts,mode)
        for blk in self.blocks: x=blk(x)
        return self.head(self.norm(x))

    def loss(self,image_ids,speed,curvature,texts=None,mode='both'):
        b,t,n=image_ids.shape; logits=self(image_ids,speed,curvature,texts,mode)
        targets=torch.full(logits.shape[:2],-100,dtype=torch.long,device=logits.device)
        for ti in range(t):
            start=ti*self.step+self.mt
            pred=torch.arange(start-1,start+n-1,device=logits.device)
            targets[:,pred]=image_ids[:,ti]
        return F.cross_entropy(logits.reshape(-1,self.K),targets.reshape(-1),ignore_index=-100)

    @staticmethod
    def sample_mode():
        r=random.random()
        return 'unconditional' if r<.2 else ('action' if r<.6 else 'text')
