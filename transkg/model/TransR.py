import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import Model

class TransR(Model):
    '''
    The TransR model implementation.
    paper title: Learning Entity and Relation Embeddings for Knowledge Graph Completion.
    paper author:Lin Y, Liu Z, Zhu X, et al.
    paper website:http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf
    '''
    def __init__(self,ent_tot,rel_tot,emb_dim,margin=1.0,L=2):
        super(TransR, self).__init__(ent_tot,rel_tot)
        assert (L==1 or L==2)
        self.name = "TransR"
        self.emb_dim = emb_dim
        self.margin = margin
        self.L = L
        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,embedding_dim=emb_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,embedding_dim=emb_dim)
        self.transfer = nn.Parameter(torch.rand(size=(emb_dim,emb_dim)),requires_grad=True)
        self.distfn = nn.PairwiseDistance(L)
    def scoreOp(self,inputTriples):
        '''
        Calculate the score,steps are as follows:
        Step1: Split input triple as head,relation and tail.
        Step2: Calculate the r-space convert vector or matrix.
        Step3: Calculate the mapping vector of head and tail.
        Step4: Return the score.
        :param inputTriples: The input vectors.
        :return: score
        '''
        # Step1
        head,relation,tail = torch.chunk(input=inputTriples,chunks=3,dim=1)
        # Step2
        head = torch.squeeze(self.entEmbedding(head),dim=1)
        relation = torch.squeeze(self.relEmbedding(relation),dim=1)
        tail = torch.squeeze(self.entEmbedding(tail),dim=1)
        head = torch.matmul(head,self.transfer)
        tail = torch.matmul(tail,self.transfer)
        # Step3 and Step4
        score = self.distfn(head+relation,tail)
        return score
    def forward(self,posX,negX):
        size = posX.size()[0]
        # Calculate the score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)
        # Use the margin ranking loss: max(posScore-negScore+margin,0)
        return torch.sum(F.relu(input=posScore-negScore+self.margin))/size
    def predict(self,inputTriples):
        return self.scoreOp(inputTriples)
    def normalizeEmbedding(self):
        '''
        In every training step,the entity embedding should be normalize fisrt.
        Here are the steps for normalizing the embedding.
        Step1: Get numpy.array from embedding weight.
        Step2: Normalize array.
        Step3: Assign normalized array to embedding.
        :return:
        '''
        weight = self.entEmbedding.weight.detach().cpu().numpy()
        weight = weight/np.sqrt(np.sum(np.square(weight),axis=1,keepdims=True))
        self.entEmbedding.weight.data.copy_(torch.from_numpy(weight))
    def retEvalWeights(self):
        '''
        Get the embedding weight from model.
        :return:
        '''
        return {"entEmbedding":self.entEmbedding.weight.detach().cpu().numpy(),
                "relEmbedding":self.relEmbedding.weight.detach().cpu().numpy(),
                "transfer":self.transfer.detach().cpu().numpy()}
    def initialWeight(self,filename):
        embeddings = np.load(filename, allow_pickle=True)
        self.entEmbedding.weight.data.copy_(embeddings["entEmbedding"])
        self.relEmbedding.weight.data.copy_(embeddings["relEmbedding"])
        self.transfer.data.copy_(embeddings["transfer"])
