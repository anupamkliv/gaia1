import torch
from gaia1.tokenizer.model import ImageTokenizer
from gaia1.diffusion.schedule import CosineSchedule

def test_tokenizer():
    m=ImageTokenizer(128,32,16,(1,2,2,2,2),1,16,.25); x=torch.randn(1,3,288,512); o=m(x); assert o['ids'].shape==(1,18,32)

def test_schedule():
    s=CosineSchedule(100); x=torch.randn(2,3,7,8,8); n=torch.randn_like(x); t=torch.tensor([2,5]); assert s.add_noise(x,n,t).shape==x.shape
