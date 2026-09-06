import argparse,torch
from torch.utils.data import DataLoader
from gaia1.utils.config import load_config
from gaia1.data.dataset import DrivingSequence,collate
from gaia1.tokenizer.model import ImageTokenizer
from gaia1.diffusion.model import VideoDiffusionDecoder
from gaia1.diffusion.schedule import CosineSchedule
from gaia1.diffusion.tasks import sample_task,masks,diffusion_loss

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--tokenizer_ckpt',required=True); p.add_argument('--out',default='decoder.pt'); p.add_argument('--steps',type=int,default=1000); a=p.parse_args()
    c=load_config(a.config); d='cuda' if torch.cuda.is_available() else 'cpu'; tc=c.tokenizer; dc=c.video_decoder
    tok=ImageTokenizer(tc.codebook_size,tc.embed_dim,tc.base_channels,tuple(tc.channel_mults),tc.num_res_blocks,tc.projection_dim,tc.commitment_beta).to(d)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt,map_location=d)['model']); tok.eval()
    for q in tok.parameters(): q.requires_grad=False
    m=VideoDiffusionDecoder(tc.codebook_size,dc.token_embed_dim,dc.base_channels,dc.num_heads).to(d); sch=CosineSchedule(dc.diffusion_steps).to(d)
    dl=DataLoader(DrivingSequence(a.manifest,7,(c.image.height,c.image.width)),batch_size=1,shuffle=True,collate_fn=collate); it=iter(dl)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=.01,betas=(.9,.99)); ema={k:v.detach().clone() for k,v in m.state_dict().items() if torch.is_floating_point(v)}
    for step in range(a.steps):
        try: b=next(it)
        except StopIteration: it=iter(dl); b=next(it)
        frames=b['frames'].to(d); B,T=frames.shape[:2]; x0=frames.permute(0,2,1,3,4)
        with torch.no_grad(): ids=tok.tokenize(frames.flatten(0,1)).view(B,T,-1)
        task=sample_task(); context,token_mask=masks(task,B,T,d); predict=1-context
        ti=torch.randint(0,sch.steps,(B,),device=d); noise=torch.randn_like(x0); xt=sch.add_noise(x0,noise,ti)
        xt=xt*predict[:,None,:,None,None]+x0*context[:,None,:,None,None]; target=sch.v_target(x0,noise,ti)
        pred=m(xt,ti,ids,token_mask,temporal_enabled=(task!='image')); loss=diffusion_loss(pred,target,predict)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        with torch.no_grad():
            for k,v in m.state_dict().items():
                if k in ema: ema[k].mul_(.999).add_(v,alpha=.001)
        if step%50==0: print(step,task,float(loss)); torch.save({'model':m.state_dict(),'ema':ema,'step':step},a.out)
if __name__=='__main__': main()
