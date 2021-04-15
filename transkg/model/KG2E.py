import numpy as np
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
    def __init__(self,ent_tot,rel_tot,emb_dim,margin=1.0,sim="KL",vmin=0.03,vmax=3.0):
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
        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,embedding_dim=emb_dim)
        self.entCovar = nn.Embedding(num_embeddings=ent_tot,embedding_dim=emb_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,embedding_dim=emb_dim)
        self.relCovar = nn.Embedding(num_embeddings=rel_tot,embedding_dim=emb_dim)
    def KLScore(self,**kargs):
        '''
        calculate the KL loss between T-H distribution and R distribution.
        There are four parts in loss function.
        :param kargs:
        :return:
        '''
        # Step1: calculate the KL(e,r)
        losep1 = torch.sum(kargs["errorv"]/kargs["relationv"],dim=1)
        losep2 = torch.sum((kargs["relationm"]-kargs["errorm"])**2/kargs["relationv"],dim=1)
        KLer = (losep1+losep2-self.emb_dim)
        # Step2: calculate the KL(r,e)
        losep1 = torch.sum(kargs["relationv"]/kargs["errorv"],dim=1)
        losep2 = torch.sum((kargs["errorm"]-kargs["relationm"])**2/kargs["errorv"],dim=1)
        KLre = (losep1+losep2-self.emb_dim)/2
        return (KLre+KLer)/2
    def ELScore(self,**kargs):
        '''
        calculate the EL loss between T-H distribution and R distribution.
        There are three parts in loss function.1
        :param kargs:
        :return:
        '''
        losep1 = torch.sum((kargs["errorm"]-kargs["relationm"])**2/(kargs["errorv"]+kargs["relationv"]),dim=1)
        losep2 = torch.sum(torch.log(kargs["errorv"]+kargs["relationv"]),dim=1)
        return (losep1+losep2)/2
    def scoreOp(self,inputTriples):
        '''
        Calculate the score of triples.
        Step1: Split the input as head,relation and tail index.
        Step2: Transform index tensor to embedding.
        Step3: Calculate the score with "KL" or "EL".
        Step4: Return the score.
        :param inputTriples:
        :return:
        '''
        head,relation,tail = torch.chunk(input=inputTriples,chunks=3,dim=1)
        headm = torch.squeeze(self.entEmbedding(head),dim=1)
        headv = torch.squeeze(self.entCovar(head),dim=1)
        tailm = torch.squeeze(self.entEmbedding(tail),dim=1)
        tailv = torch.squeeze(self.entCovar(tail),dim=1)
        relationm = torch.squeeze(self.relEmbedding(relation),dim=1)
        relationv = torch.squeeze(self.relCovar(relation),dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        if self.sim == "KL":
            return self.KLScore(relationm=relationm,relationv=relationv,errorv=errorv,errorm=errorm)
        elif self.sim == "EL":
            return self.ELScore(relationm=relationm,relationv=relationv,errorm=errorm,errorv=errorv)
        else:
            print("ERROR : Sim %s is not supported!"%self.sim)
            exit(1)
    def normalizeEmbedding(self):
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
        self.entCovar.weight.data.copy_(torch.clamp(input=self.entCovar.weight.detach().cpu(),
                                                       min=self.vmin,
                                                       max=self.vmax))
        self.relCovar.weight.data.copy_(torch.clamp(input=self.relCovar.weight.detach().cpu(),
                                                         min=self.vmin,
                                                         max=self.vmax))
    def forward(self,posX,negX):
        size = posX.size()[0]

        # calculate the score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)

        return torch.sum(F.relu(input=posScore-negScore+self.margin))/size
    def retEvalWeights(self):
        '''
        Return the KG2E model embedding.
        :return:
        '''
        return {"entEmbedding":self.entEmbedding.weight.detach().cpu().numpy(),
                "relEmbedding":self.relEmbedding.weight.detach().cpu().numpy(),
                "entCovar":self.entCovar.weight.detach().cpu().numpy(),
                "relCovar":self.relCovar.weight.detach().cpu().numpy(),
                "sim":self.sim}
    def initialWeight(self,filename):
        embeddings = np.load(filename, allow_pickle=True)
        self.entEmbedding.weight.data.copy_(embeddings["entEmbedding"])
        self.relEmbedding.weight.data.copy_(embeddings["relEmbedding"])
        self.entCovar.weight.data.copy_(embeddings["entCovar"])
        self.relCovar.weight.data.copy_(embeddings["relCovar"])
        self.sim = embeddings["sim"]
    def predictSimScore(self,head,relation,simMeasure="dot"):
        pass