import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import functional as TF

def image(path,size=(288,512)):
    im=Image.open(path).convert('RGB').resize((size[1],size[0])); return TF.to_tensor(im)*2-1

class ImageFolder(Dataset):
    def __init__(self,root,size=(288,512)):
        self.f=[p for p in Path(root).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]; self.size=size
    def __len__(self): return len(self.f)
    def __getitem__(self,i): return image(self.f[i],self.size)

class DrivingSequence(Dataset):
    def __init__(self,manifest,frames=26,size=(288,512)):
        self.rows=[json.loads(x) for x in open(manifest,encoding='utf-8') if x.strip()]; self.frames=frames; self.size=size
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; paths=r['frames'][:self.frames]; texts=r.get('texts',['']*len(paths)); texts=[texts]*len(paths) if isinstance(texts,str) else texts
        return {'frames':torch.stack([image(p,self.size) for p in paths]),'texts':texts[:self.frames],
                'speed':torch.tensor(r['speed'][:self.frames],dtype=torch.float32),
                'curvature':torch.tensor(r['curvature'][:self.frames],dtype=torch.float32)}

def collate(batch):
    return {'frames':torch.stack([x['frames'] for x in batch]),'texts':[x['texts'] for x in batch],
            'speed':torch.stack([x['speed'] for x in batch]),'curvature':torch.stack([x['curvature'] for x in batch])}
