import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16,VGG16_Weights

class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
        for p in self.net.parameters(): p.requires_grad=False
    def forward(self,x,y):
        x=(x+1)/2; y=(y+1)/2
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
        std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        return F.l1_loss(self.net((x-mean)/std),self.net((y-mean)/std))

class PatchDiscriminator(nn.Module):
    def __init__(self,base=64):
        super().__init__(); layers=[nn.Conv2d(3,base,4,2,1),nn.LeakyReLU(.2)]
        c=base
        for m in (2,4,8):
            layers += [nn.Conv2d(c,base*m,4,2,1),nn.LeakyReLU(.2)]; c=base*m
        layers += [nn.Conv2d(c,1,3,padding=1)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def d_hinge(real,fake): return .5*((1-real).relu().mean()+(1+fake).relu().mean())
def g_hinge(fake): return -fake.mean()

class DinoAlignment(nn.Module):
    def __init__(self,quant_dim,teacher_dim):
        super().__init__(); self.proj=nn.Conv2d(quant_dim,teacher_dim,1)
    def forward(self,q,t):
        q=self.proj(q)
        t=F.interpolate(t,size=q.shape[-2:],mode='bilinear',align_corners=False)
        return (1-(F.normalize(q,dim=1)*F.normalize(t.detach(),dim=1)).sum(1)).mean()
