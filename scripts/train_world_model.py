import argparse,torch
from torch.utils.data import DataLoader
from gaia1.utils.config import load_config
from gaia1.data.dataset import DrivingSequence,collate
from gaia1.tokenizer.model import ImageTokenizer
from gaia1.world_model.model import GaiaWorldModel

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--manifest',required=True); p.add_argument('--tokenizer_ckpt',required=True); p.add_argument('--out',default='world.pt'); p.add_argument('--steps',type=int,default=1000); a=p.parse_args()
    c=load_config(a.config); d='cuda' if torch.cuda.is_available() else 'cpu'; tc=c.tokenizer; wc=c.world_model
    tok=ImageTokenizer(tc.codebook_size,tc.embed_dim,tc.base_channels,tuple(tc.channel_mults),tc.num_res_blocks,tc.projection_dim,tc.commitment_beta).to(d)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt,map_location=d)['model']); tok.eval()
    for q in tok.parameters(): q.requires_grad=False
    m=GaiaWorldModel(wc.codebook_size,wc.hidden_dim,wc.num_layers,wc.num_heads,wc.ffn_mult,wc.max_timesteps,wc.text_tokens,wc.image_tokens,wc.action_tokens,wc.t5_name,wc.freeze_t5,wc.dropout).to(d)
    dl=DataLoader(DrivingSequence(a.manifest,wc.max_timesteps,(c.image.height,c.image.width)),batch_size=1,shuffle=True,collate_fn=collate); it=iter(dl)
    opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=1e-4,weight_decay=.1,betas=(.9,.95))
    for step in range(a.steps):
        try: b=next(it)
        except StopIteration: it=iter(dl); b=next(it)
        f=b['frames'].to(d); B,T=f.shape[:2]
        with torch.no_grad(): ids=tok.tokenize(f.flatten(0,1)).view(B,T,-1)
        loss=m.loss(ids,b['speed'].to(d),b['curvature'].to(d),b['texts'],m.sample_mode())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step()
        if step%50==0: print(step,float(loss)); torch.save({'model':m.state_dict(),'step':step},a.out)
if __name__=='__main__': main()
