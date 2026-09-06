import argparse,torch
from gaia1.utils.config import load_config
from gaia1.tokenizer.model import ImageTokenizer
from gaia1.world_model.model import GaiaWorldModel
from gaia1.world_model.generation import generate_next_frame
from gaia1.diffusion.model import VideoDiffusionDecoder
from gaia1.diffusion.schedule import CosineSchedule
from gaia1.diffusion.sampling import ddim_sample

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--tokenizer_ckpt',required=True)
    p.add_argument('--world_ckpt',required=True); p.add_argument('--decoder_ckpt',required=True); p.add_argument('--input_clip',required=True)
    p.add_argument('--out',default='gaia1_rollout.pt'); a=p.parse_args()
    c=load_config(a.config); d='cuda' if torch.cuda.is_available() else 'cpu'; tc=c.tokenizer; wc=c.world_model; dc=c.video_decoder

    tok=ImageTokenizer(tc.codebook_size,tc.embed_dim,tc.base_channels,tuple(tc.channel_mults),tc.num_res_blocks,tc.projection_dim,tc.commitment_beta).to(d)
    tok.load_state_dict(torch.load(a.tokenizer_ckpt,map_location=d)['model']); tok.eval()
    world=GaiaWorldModel(wc.codebook_size,wc.hidden_dim,wc.num_layers,wc.num_heads,wc.ffn_mult,wc.max_timesteps,wc.text_tokens,wc.image_tokens,wc.action_tokens,wc.t5_name,wc.freeze_t5,wc.dropout).to(d)
    world.load_state_dict(torch.load(a.world_ckpt,map_location=d)['model']); world.eval()
    decoder=VideoDiffusionDecoder(tc.codebook_size,dc.token_embed_dim,dc.base_channels,dc.num_heads).to(d)
    decoder.load_state_dict(torch.load(a.decoder_ckpt,map_location=d)['model']); decoder.eval()

    sample=torch.load(a.input_clip,map_location=d)
    frames=sample['frames']; T=frames.shape[0]
    with torch.no_grad(): ids=tok.tokenize(frames).unsqueeze(0)
    ids=torch.cat([ids,torch.zeros(1,1,576,dtype=torch.long,device=d)],1)
    speed=sample['speed'][:T+1].unsqueeze(0).to(d); curvature=sample['curvature'][:T+1].unsqueeze(0).to(d)
    texts=sample.get('texts',['']*(T+1)); texts=[texts[:T+1]]
    ids[:,-1]=generate_next_frame(world,ids,speed,curvature,texts,k=50,temp=1.,mode='both')

    # Decode a short joint window. Long sequences should use overlapping sliding windows.
    window=ids[:,-7:]; sch=CosineSchedule(dc.diffusion_steps).to(d)
    rgb=ddim_sample(decoder,sch,window,(1,3,window.shape[1],c.image.height,c.image.width),d,steps=50,
                    token_mask=torch.ones(1,window.shape[1],device=d))
    torch.save({'video':rgb.cpu(),'tokens':window.cpu()},a.out)

if __name__=='__main__': main()
