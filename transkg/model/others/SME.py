import torch
import torch.nn as nn

from ..Model import Model

class SME(Model):
    def __init__(self,ent_tot,rel_tot,ent_dim,rel_dim,bilinear = False):
        super(SME, self).__init__(ent_tot,rel_tot)
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim

        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=ent_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                         embedding_dim=rel_dim)
        if bilinear:
            self.LinearR1 = nn.Linear(ent_dim,rel_dim)
        self.LinearR = nn.Bilinear(ent_dim,ent_dim,rel_dim,bias=False)
    def scoreOp(self):
        '''
        Calculate the score of the SME model.
        :return:
        '''
    def forward(self,posX,negX):
        pass
    def predict(self, *args, **kwargs):
        pass