import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import Model

class TransD(Model):
    '''
    The TransD model implementation.
    paper title: Knowledge Graph Embedding via Dynamic Mapping Matrix.
    paper author: Ji G , He S , Xu L , et al.
    paper website: http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Knowledge%20Graph%20Embedding%20via%20Dynamic%20Mapping%20Matrix.pdf
    '''
    def __init__(self,ent_tot,rel_tot,ent_dim,rel_dim,margin=1.0,L=2):
        super(TransD, self).__init__(ent_tot,rel_tot)
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.margin = margin
        self.L = L
        # Initialize the entity and relation embedding and projection embedding
        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=ent_dim)
        self.entMapEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                            embedding_dim=ent_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                         embedding_dim=rel_dim)
        self.relMapEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                            embedding_dim=rel_dim)
        self.distfn = nn.PairwiseDistance(L)
    def scoreOp(self,inputTriples):
        '''
        Calculate the score,steps are as follows:
        Step1: Split input triple as head,relation adn tail.
        Step2: Calculate the mapping matrix Mrh and Mrt.
        Step3: Calculate the mapping vector of head and tail.
        Step4: Return the score.
        :return:
        '''
        head,relation,tail = torch.chunk(input=inputTriples,chunks=3,dim=1)
        headp = torch.squeeze(self.entMapEmbedding(head),dim=1)
        head = torch.squeeze(self.entEmbedding(head),dim=1)
        relationp = torch.squeeze(self.relMapEmbedding(relation),dim=1)
        relation = torch.squeeze(self.relEmbedding(relation),dim=1)
        tailp = torch.squeeze(self.entMapEmbedding(tail),dim=1)
        tail = torch.squeeze(self.entEmbedding(tail),dim=1)
        relationp = torch.unsqueeze(relationp,dim=2)
        headp = torch.unsqueeze(headp,dim=1)
        tailp = torch.unsqueeze(tailp,dim=1)
        if inputTriples.is_cuda:
            Mrh = torch.matmul(relationp,headp).cuda(inputTriples.device.index)+torch.eye(self.rel_dim,self.ent_dim).cuda(inputTriples.device.index)
            Mrt = torch.matmul(relationp,tailp).cuda(inputTriples.device.index) + torch.eye(self.rel_dim, self.ent_dim).cuda(inputTriples.device.index)
        else:
            Mrh = torch.matmul(relationp,headp)+torch.eye(self.rel_dim,self.ent_dim)
            Mrt = torch.matmul(relationp, tailp) + torch.eye(self.rel_dim, self.ent_dim)
        head = torch.unsqueeze(head,dim=2)
        tail = torch.unsqueeze(tail,dim=2)
        head = torch.squeeze(torch.matmul(Mrh,head),dim=2)
        tail = torch.squeeze(torch.matmul(Mrt,tail),dim=2)
        return self.distfn(head+relation,tail)
    def normalizeEmbedding(self):
        '''
        normalize the embedding and change the embedding.
        :return:
        '''
        self.entEmbedding.weight.data.copy_(torch.renorm(input=self.entMapEmbedding.weight.detach().cpu(),
                                                            p=2,
                                                            dim=0,
                                                            maxnorm=1))
        self.relEmbedding.weight.data.copy_(torch.renorm(input=self.relEmbedding.weight.detach().cpu(),
                                                              p=2,
                                                              dim=0,
                                                              maxnorm=1))
    def retEvalWeights(self):
        '''
        Return the weights of the TransD model.
        :return:
        '''
        return {"entityEmbedding": self.entEmbedding.weight.detach().cpu().numpy(),
                "relationEmbedding": self.relEmbedding.weight.detach().cpu().numpy(),
                "entityMapEmbedding": self.entMapEmbedding.weight.detach().cpu().numpy(),
                "relationMapEmbedding": self.relMapEmbedding.weight.detach().cpu().numpy()}
    def forward(self,posX,negX):
        size = posX.size()[0]
        # Calculate the score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)
        return torch.sum(F.relu(input=posScore-negScore+self.margin))/size
    def predict(self, *args, **kwargs):
        pass