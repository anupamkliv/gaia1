import torch
import torch.nn as nn

class DinoTeacher(nn.Module):
    '''
    Adapter for an official DINO ViT loaded by the user.

    Example:
        dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        teacher = DinoTeacher(dino, patch_size=16)

    GAIA-1 states that DINO features are distilled with cosine similarity, but the
    paper does not disclose the exact DINO variant or selected feature layer.
    '''
    def __init__(self,model,patch_size=16):
        super().__init__(); self.model=model.eval(); self.patch_size=patch_size
        for p in self.model.parameters(): p.requires_grad=False

    @torch.no_grad()
    def forward(self,x):
        z=self.model.get_intermediate_layers(x,n=1)[0]
        h=x.shape[-2]//self.patch_size; w=x.shape[-1]//self.patch_size
        if z.shape[1]==h*w+1: z=z[:,1:]
        z=z[:,:h*w]
        return z.transpose(1,2).reshape(x.shape[0],z.shape[-1],h,w)
