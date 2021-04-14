import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import Model

class TransA(Model):
    '''
    The implementation of TransA model.
    paper title: TransA: An Adaptive Approach for Knowledge Graph Embedding
    paper author: Xiao H ,  Huang M ,  Hao Y , et al.
    paper website: https://arxiv.org/pdf/1509.05490.pdf
    '''
    def __init__(self,ent_tot,rel_tot,emb_dim,margin=1.0,L=2,lamb=0.01,C=0.2):
        super(TransA, self).__init__(ent_tot,rel_tot)
        assert (L==1 or L==2)
        self.name = "TransA"
        self.emb_dim = emb_dim
        self.margin = margin
        self.lamb = lamb
        self.C = C

        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=emb_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                         embedding_dim=emb_dim)
        self.relWeight = nn.Parameter(torch.rand(size=(rel_tot,emb_dim,emb_dim)),requires_grad=True)
        self.distfn = nn.PairwiseDistance(L)
    def scoreOp(self,inputTriples):
        '''
        Calculate score,steps follows:
        Step1: Split input as head,relation,tail.
        Step2: Transform index array to embedding vector.
        Step3: Calculate Mahalanobis distance weights.
        Step4: Calculate distance as final score.
        :return:
        '''
        head,relation,tail = torch.chunk(input=inputTriples,chunks=3,dim=1)
        relWr = self.relWeight[relation]
        head = torch.squeeze(self.entEmbedding(head),dim=1)
        relation = torch.squeeze(self.relEmbedding(relation),dim=1)
        tail = torch.squeeze(self.entEmbedding(tail),dim=1)
        error = torch.unsqueeze(torch.abs(head+relation-tail),dim=1)
        error = torch.matmul(torch.matmul(error,torch.unsqueeze(relWr,dim=0)),error.permute(0,2,1))
        return torch.squeeze(error)
    def resetWr(self):
        '''
        Clean the relation weight.
        :return:
        '''
        self.relWeight.data.copy_(torch.zeros(size=(self.rel_tot,self.emb_dim,self.emb_dim)))
    def calculateWr(self,posX,negX):
        '''
        :param posX:
        :param negX:
        :return:
        '''
        posHead,posRel,posTail = torch.chunk(input=posX,chunks=3,dim=1)
        negHead,negRel,negTail = torch.chunk(input=negX,chunks=3,dim=1)
        posHeadM, posRelM, posTailM = self.entEmbedding(posHead),\
                                      self.relEmbedding(posRel),\
                                      self.entEmbedding(posTail)
        negHeadM, negRelM, negTailM = self.entEmbedding(negHead), \
                                      self.relEmbedding(negRel), \
                                      self.entEmbedding(negTail)
        errorPos = torch.abs(posHeadM+posRelM-posTailM)
        errorNeg = torch.abs(negHeadM+negRelM-negTailM)
        del posHeadM,posRelM,posTailM,negHeadM,negRelM,negTailM
        self.relWeight.data[posRel] += torch.sum(torch.matmul(errorNeg.permute((0, 2, 1)), errorNeg),dim=0) - \
                                               torch.sum(torch.matmul(errorPos.permute((0, 2, 1)), errorPos), dim=0)

    def forward(self,posX,negX):
        '''
        Calculate the score.
        :param posX:
        :param negX:
        :return:
        '''
        size = posX.size()[0]
        self.calculateWr(posX,negX)
        # Calculate the score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)
        # Calculate loss
        marginLoss = 1/size * torch.sum(F.relu(input=posScore-negScore+self.margin))
        WrLoss = 1/size * torch.norm(input=self.relWeight.data,p=self.L)
        WLoss = 1/self.ent_tot * torch.norm(input=self.entEmbedding.weight,p=2)+\
            1/self.rel_tot * torch.norm(input=self.relEmbedding.weight,p=2)
        return marginLoss+WrLoss+WLoss
    def normalizeEmbedding(self):
        '''
        Normalize the embedding.
        :return:
        '''
        '''
                Normalize the embedding.
                :return:
                '''
        self.entEmbedding.weight.data.copy_(torch.renorm(input=self.entEmbedding.weight.detach().cpu(),
                                                         p=2,
                                                         dim=0,
                                                         maxnorm=1.0))
        self.relEmbedding.weight.data.copy_(torch.renorm(input=self.relEmbedding.weight.detach().cpu(),
                                                         p=2,
                                                         dim=0,
                                                         maxnorm=1.0))
        self.relWeight.data.copy_(torch.renorm(input=self.relWeight.data.detach().cpu(),
                                                         p=2,
                                                         dim=0,
                                                         maxnorm=1.0))
    def predict(self, inputTriples):
        return self.scoreOp(inputTriples)
    def retEvalWeights(self):
        '''
        Return the embedding of the model.
        :return:
        '''
        return {"entEmbedding":self.entEmbedding.weight.detach().cpu().numpy(),
                "relEmbedding":self.relEmbedding.weight.detach().cpu().numpy(),
                "relWeight":self.relWeight.detach().cpu().numpy()}
    def initialWeight(self,filename):
        embeddings = np.load(filename, allow_pickle=True)
        self.entEmbedding.weight.data.copy_(embeddings["entEmbedding"])
        self.relEmbedding.weight.data.copy_(embeddings["relEmbedding"])
        self.relWeight.data.copy_(embeddings["relWeight"])

