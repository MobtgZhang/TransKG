import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Model import Model

class SME(Model):
    def __init__(self,ent_tot,rel_tot,ent_dim,rel_dim,L=2,margin=1.0,ele_dot = False):
        super(SME, self).__init__(ent_tot,rel_tot)
        assert (L==1 or L==2)
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.ele_dot = ele_dot
        self.L = L
        self.margin = margin
        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=ent_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                         embedding_dim=rel_dim)
        if ele_dot:
            self.LinearR1 = nn.Linear(ent_dim, 1, bias=False)
            self.LinearR2 = nn.Linear(rel_dim, 1, bias=False)
            self.LinearR3 = nn.Linear(ent_dim, 1, bias=False)
            self.LinearR4 = nn.Linear(rel_dim, 1, bias=False)
            self.bias1 = nn.Parameter(torch.Tensor(1),requires_grad=True)
            self.bias2 = nn.Parameter(torch.Tensor(1),requires_grad=True)
        else:
            self.LinearR1 = nn.Linear(ent_dim, 1, bias=False)
            self.LinearR2 = nn.Linear(rel_dim, 1)
            self.LinearR3 = nn.Linear(ent_dim, 1, bias=False)
            self.LinearR4 = nn.Linear(rel_dim, 1)
        self.distfn = nn.PairwiseDistance(L)
    def scoreOp(self,inputTriples):
        '''
        Calculate the score of the SME model.
        :return:
        '''
        head, relation, tail = torch.chunk(input=inputTriples, chunks=3, dim=1)
        head = torch.squeeze(self.entEmbedding(head), dim=1)
        tail = torch.squeeze(self.entEmbedding(tail), dim=1)
        relation = torch.squeeze(self.relEmbedding(relation), dim=1)

        if self.ele_dot:
            left = self.LinearR1(head) * self.LinearR2(relation) + self.bias1
            right = self.LinearR3(head) * self.LinearR4(relation) + self.bias2
        else:
            left = self.LinearR1(head)+self.LinearR2(relation)
            right = self.LinearR3(tail)+self.LinearR4(relation)
        score = torch.squeeze(left*right)
        return score
    def normalizeEmbedding(self):
        '''
        In every training epoch,the entity embedding should be normalize first.\\
        There are three steps:
        ==> Step1: Get numpy.array from embedding weight.
        ==> Step2: Normalize array.
        ==> Step3: Assign normalized array to embedding.
        :return:
        '''
        weight = self.entEmbedding.weight.detach().cpu().numpy()
        weight = weight / np.sqrt(np.sum(np.square(weight), axis=1, keepdims=True))
        self.entEmbedding.weight.data.copy_(torch.from_numpy(weight))
        weight = self.relEmbedding.weight.detach().cpu().numpy()
        weight = weight / np.sqrt(np.sum(np.square(weight), axis=1, keepdims=True))
        self.relEmbedding.weight.data.copy_(torch.from_numpy(weight))
    def forward(self,posX,negX):
        '''
        Use positive and negative data to calculate the training score.
        :param posX: positive data
        :param negX: negative data
        :return:
        '''
        size = posX.size()[0]
        # Calculate the score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        # Get Margin ranking loss: max(posScore-negScore+margin,0)
        return torch.sum(F.relu(input=posScore - negScore + self.margin)) / size
    def predict(self,inputTriples):
        return self.scoreOp(inputTriples)
    def retEvalWeights(self):
        if self.ele_dot:
            return {"entityEmbedding": self.entEmbedding.weight.detach().cpu().numpy(),
                    "relationEmbedding": self.relEmbedding.weight.detach().cpu().numpy(),
                    "M1": self.LinearR1.weight.detach().cpu().numpy(),
                    "M2": self.LinearR2.weight.detach().cpu().numpy(),
                    "M3": self.LinearR3.weight.detach().cpu().numpy(),
                    "M4": self.LinearR4.weight.detach().cpu().numpy(),
                    "b1": self.bias1.detach().cpu().numpy(),
                    "b2": self.bias2.detach().cpu().numpy(),
                    "ele_tot":True}
        else:
            return {"entityEmbedding":self.entEmbedding.weight.detach().cpu().numpy(),
                    "relationEmbedding":self.relEmbedding.weight.detach().cpu().numpy(),
                    "M1":self.LinearR1.weight.detach().cpu().numpy(),
                    "M2":self.LinearR2.weight.detach().cpu().numpy(),
                    "M3":self.LinearR3.weight.detach().cpu().numpy(),
                    "M4":self.LinearR4.weight.detach().cpu().numpy(),
                    "b1":self.LinearR1.bias.detach().cpu().numpy(),
                    "b2":self.LinearR2.bias.detach().cpu().numpy(),
                    "ele_tot":False}
