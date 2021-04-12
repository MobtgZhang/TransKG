import torch
import torch.nn.functional as F

from .Model import Model

class TransR(Model):
    def __init__(self,ent_tot,rel_tot):
        super(TransR, self).__init__(ent_tot,rel_tot)
    def forward(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass
