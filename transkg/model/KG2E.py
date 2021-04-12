import torch
import torch.nn.functional as F

from .Model import Model

class KG2E(Model):
    def __init__(self,ent_tot,rel_tot):
        super(KG2E, self).__init__(ent_tot,rel_tot)
    def forward(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass

