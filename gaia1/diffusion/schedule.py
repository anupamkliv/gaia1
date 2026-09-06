import math
import torch

class CosineSchedule:
    def __init__(self,steps=1000,s=.008):
        self.steps=steps; x=torch.linspace(0,steps,steps+1); abar=torch.cos(((x/steps+s)/(1+s))*math.pi/2)**2; abar/=abar[0]
        self.betas=(1-abar[1:]/abar[:-1]).clamp(1e-5,.999); self.alphas=1-self.betas; self.abar=torch.cumprod(self.alphas,0)
    def to(self,d): self.betas=self.betas.to(d); self.alphas=self.alphas.to(d); self.abar=self.abar.to(d); return self
    def _a_s(self,t): return self.abar[t].sqrt()[:,None,None,None,None],(1-self.abar[t]).sqrt()[:,None,None,None,None]
    def add_noise(self,x,n,t): a,s=self._a_s(t); return a*x+s*n
    def v_target(self,x,n,t): a,s=self._a_s(t); return a*n-s*x
    def x0_from_v(self,xt,v,t): a,s=self._a_s(t); return a*xt-s*v
    def eps_from_v(self,xt,v,t): a,s=self._a_s(t); return s*xt+a*v
