import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import Model

class KG2E(Model):
    '''
    The implementation for KG2E model.
    paper title: Learning to Represent Knowledge Graphs with Gaussian Embedding.
    paper author: He S , Kang L , Ji G , et al.
    paper website: http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf
    '''
    def __init__(self,ent_tot,rel_tot,ent_dim,rel_dim,emb_dim,margin=1.0,sim="KL",vmin=0.03,vmax=3.0):
        super(KG2E, self).__init__(ent_tot,rel_tot)
        assert (sim in ["KL","EL"])
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.name = "KG2E"
        self.margin = margin
        self.sim = sim
        self.emb_dim = emb_dim
        self.vmin = vmin
        self.vmax = vmax

        # Embeddings represent the mean vector of entity and relation
        # Covars representation the covariance vector of entity and relation
        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,embedding_dim=ent_dim)
        self.entCovar = nn.Embedding(num_embeddings=ent_tot,embedding_dim=ent_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,embedding_dim=rel_dim)
        self.relCovar = nn.Embedding(num_embeddings=rel_tot,embedding_dim=rel_dim)
    def KLScore(self,**kargs):
        pass
    def ELScore(self,**kargs):
        pass
    def forward(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass

