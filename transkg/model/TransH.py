import torch
import torch.nn.functional as F

from .Model import Model

class TransH(Model):
    def __init__(self,ent_tot,rel_tot):
        super(TransH, self).__init__(ent_tot,rel_tot)
    def forward(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass
