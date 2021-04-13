import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..Model import Model

class NTN(Model):
    def __init__(self,ent_tot,rel_tot,ent_dim,rel_dim,margin=1.0,bias_flag = True,rel_flag = True):
        super(NTN, self).__init__(ent_tot,rel_tot)
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.rel_flag = rel_flag
        self.bias_flag = bias_flag
        self.margin = margin

        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=ent_dim)

        self.LinearBR = nn.Bilinear(ent_dim,ent_dim,rel_dim,bias=False)
        self.LinearRH = nn.Linear(ent_dim,rel_dim,bias=False)
        self.LinearRT = nn.Linear(ent_dim,rel_dim,bias=False)
        if rel_flag:
            self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                             embedding_dim=rel_dim)
            self.LinearRR = nn.Linear(rel_dim,rel_dim,bias=False)
        if bias_flag:
            self.bias = nn.Parameter(torch.rand(size=(rel_dim,1)),requires_grad=True)
        self.UW = nn.Parameter(torch.rand(size=(rel_dim,1)),requires_grad=True)

    def scoreOp(self,inputTriples):
        '''
        :param inputTriples:
        :return:
        '''
        head, relation, tail = torch.chunk(input=inputTriples, chunks=3, dim=1)
        head = torch.squeeze(self.entEmbedding(head), dim=1)
        tail = torch.squeeze(self.entEmbedding(tail), dim=1)
        if self.rel_flag:
            relation = torch.squeeze(self.relEmbedding(relation), dim=1)
            predict = self.LinearBR(head,tail)+self.LinearRH(head)+self.LinearRT(tail)+self.LinearRR(relation)
        else:
            predict = self.LinearBR(head, tail) + self.LinearRH(head) + self.LinearRT(tail)
        score = torch.squeeze(torch.mm(torch.tanh(predict), self.UW))
        return score
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
        returnDict = {"entityEmbedding": self.entEmbedding.weight.detach().cpu().numpy(),
                      "BR_w":self.LinearBR.weight.detach().cpu().numpy(),
                      "RH_w":self.LinearRH.weight.detach().cpu().numpy(),
                      "RT_w":self.LinearRH.weight.detach().cpu().numpy()}
        if self.rel_flag:
            returnDict.update({"relationEmbedding": self.relEmbedding.weight.detach().cpu().numpy(),
                      "RR_w":self.LinearRR.weight.detach().cpu().numpy()})
        if self.bias_flag:
            returnDict.update({"Bias":self.bias.detach().cpu().numpy()})
        return returnDict
    def normalizeEmbedding(self):
        '''
        Method for normalizing embedding.
        :return:
        '''
        weight = self.entEmbedding.weight.detach().cpu().numpy()
        weight = weight / np.sqrt(np.sum(np.square(weight), axis=1, keepdims=True))
        self.entEmbedding.weight.data.copy_(torch.from_numpy(weight))
        if self.rel_flag:
            weight = self.relEmbedding.weight.detach().cpu().numpy()
            weight = weight / np.sqrt(np.sum(np.square(weight), axis=1, keepdims=True))
            self.relEmbedding.weight.data.copy_(torch.from_numpy(weight))
