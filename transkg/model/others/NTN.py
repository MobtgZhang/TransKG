import torch
import torch.nn as nn

from ..Model import Model

class NTN(Model):
    def __init__(self,ent_tot,rel_tot,ent_dim,rel_dim,Mweight = True):
        super(NTN, self).__init__(ent_tot,rel_tot)
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim

        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=ent_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                         embedding_dim=rel_dim)
        self.LinearR = nn.Bilinear(ent_dim,ent_dim,rel_dim,bias=False)
    def forward(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass