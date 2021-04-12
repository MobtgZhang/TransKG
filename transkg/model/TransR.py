import torch
import torch.nn.functional as F

from .Model import Model

class TransR(Model):
    '''
    The TransR model implementation.
    paper title: Learning Entity and Relation Embeddings for Knowledge Graph Completion.
    paper author:Lin Y, Liu Z, Zhu X, et al.
    paper website:http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
    '''
    def __init__(self,ent_tot,rel_tot):
        super(TransR, self).__init__(ent_tot,rel_tot)
    def forward(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass
