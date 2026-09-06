import argparse,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from gaia1.utils.config import load_config
from gaia1.data.dataset import ImageFolder
from gaia1.tokenizer.model import ImageTokenizer
from gaia1.tokenizer.losses import VGGPerceptual,PatchDiscriminator,d_hinge,g_hinge

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data',required=True); p.add_argument('--out',default='tokenizer.pt'); p.add_argument('--steps',type=int,default=1000); a=p.parse_args()
    c=load_config(a.config); d='cuda' if torch.cuda.is_available() else 'cpu'; tc=c.tokenizer
    m=ImageTokenizer(tc.codebook_size,tc.embed_dim,tc.base_channels,tuple(tc.channel_mults),tc.num_res_blocks,tc.projection_dim,tc.commitment_beta).to(d)
    disc=PatchDiscriminator().to(d); perc=VGGPerceptual().to(d)
    dl=DataLoader(ImageFolder(a.data,(c.image.height,c.image.width)),batch_size=4,shuffle=True,num_workers=4,drop_last=True); it=iter(dl)
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=.01,betas=(.5,.9)); dopt=torch.optim.AdamW(disc.parameters(),lr=1e-4,weight_decay=.01,betas=(.5,.9))
    for step in range(a.steps):
        try: x=next(it)
        except StopIteration: it=iter(dl); x=next(it)
        x=x.to(d); out=m(x); fake=out['reconstruction']
        ld=d_hinge(disc(x.detach()),disc(fake.detach())); dopt.zero_grad(); ld.backward(); dopt.step()
        # GAIA-1 weights: .2 L1 + 2 L2 + .1 perceptual + 1 GAN + 1 codebook + .1 DINO.
        # DINO is left as a pluggable teacher because the paper does not disclose the exact DINO variant/layer.
        loss=.2*F.l1_loss(fake,x)+2*F.mse_loss(fake,x)+.1*perc(fake,x)+g_hinge(disc(fake))+out['quantization_loss']
        opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0: print(step,float(loss)); torch.save({'model':m.state_dict(),'step':step},a.out)
if __name__=='__main__': main()
