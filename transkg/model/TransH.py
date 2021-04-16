import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Model import Model

class TransH(Model):
    '''
    This is the TransH model implementation.
    paper title: Knowledge Graph Embedding by Translating on Hyperplanes
    paper author: Zhang J .
    paper website: http://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf
    '''
    def __init__(self,ent_tot,rel_tot,emb_dim=100,margin=1.0,L=2,C=1.0,eps=0.001):
        super(TransH, self).__init__(ent_tot,rel_tot)
        self.name = "TransH"
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.emb_dim = emb_dim
        self.margin = margin
        self.L = L
        self.C = C
        self.eps = eps

        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=emb_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=rel_tot,
                                         embedding_dim=emb_dim)
        self.relHyper = nn.Embedding(num_embeddings=rel_tot,
                                          embedding_dim=emb_dim)
        self.distfn = nn.PairwiseDistance(L)
    def scoreOp(self,inputTriple):
        '''
        Calculate the score:
        Step1: Split the triple as head,relation and tail.
        Step2: Transform index tensor to embedding tensor.
        Step3: Project entity head and tail embedding to relation hyperplane.
        Step4: Calculate similarity score in relation hyperplane.
        Step5: Return the score.
        :param inputTriple: The input triples for calculating the score,includes head,relation and tail.
        :return: score
        '''
        # Step1
        head,relation,tail = torch.chunk(inputTriple,chunks=3,dim=1)
        # Step2
        head = torch.squeeze(self.entEmbedding(head),dim=1)
        relHyper = torch.squeeze(self.relHyper(relation), dim=1)
        relation = torch.squeeze(self.relEmbedding(relation),dim=1)
        tail = torch.squeeze(self.entEmbedding(tail),dim=1)

        # Step3
        head = head - relHyper * torch.sum(head*relHyper,dim=1,keepdim=True)
        tail = tail - relHyper * torch.sum(tail*relHyper,dim=1,keepdim=True)
        # Step4
        return self.distfn(head+relation,tail)
    def forward(self,posX,negX):
        '''
        :param posX: positive samples
        :param negX: negative samples
        :return:
        '''
        size = posX.size()[0]
        # calculate the score
        posScore = self.scoreOp(posX)
        negScore = self.scoreOp(negX)
        # Get margin ranking loss
        # max(posScore -negScore +margin,0)
        # Use F.relu()
        marginLoss = torch.sum(F.relu(input=posScore-negScore+self.margin))
        entityLoss = torch.sum(F.relu(torch.norm(self.entEmbedding.weight,p=2,dim=1,keepdim=False)-1))
        orthLoss = torch.sum(F.relu(torch.sum(self.relHyper.weight * self.relEmbedding.weight,dim=1,keepdim=False)/\
            torch.norm(self.relEmbedding.weight,p=2,dim=1,keepdim=False)-self.eps**2))
        return marginLoss/size + self.C*(entityLoss/self.ent_tot+orthLoss/self.rel_tot)
    def normalizeEmbedding(self):
        '''
        The normalization for embedding, including relationEmbedding,entityEmbedding,hyperEmbedding.
        :return:
        '''
        weight = self.entEmbedding.weight.detach().cpu().numpy()
        weight = weight / np.sqrt(np.sum(np.square(weight), axis=1, keepdims=True))
        self.entEmbedding.weight.data.copy_(torch.from_numpy(weight))

        weight = self.relEmbedding.weight.detach().cpu().numpy()
        weight = weight / np.sqrt(np.sum(np.square(weight), axis=1, keepdims=True))
        self.relEmbedding.weight.data.copy_(torch.from_numpy(weight))
        
        weight = self.relHyper.weight.detach().cpu().numpy()
        weight = weight/np.sqrt(np.sum(np.square(weight),axis=1,keepdims=True))
        self.relHyper.weight.data.copy_(torch.from_numpy(weight))
    def retEvalWeights(self):
        '''
        The embedding for the model to save
        :return:
        '''
        return {"entEmbedding": self.entEmbedding.weight.detach().cpu().numpy(),
                "relEmbedding": self.relEmbedding.weight.detach().cpu().numpy(),
                "relHyper": self.relHyper.weight.detach().cpu().numpy()}
    def initialWeight(self,filename):
        embeddings = np.load(filename, allow_pickle=True)
        self.entEmbedding.weight.data.copy_(embeddings["entEmbedding"])
        self.relEmbedding.weight.data.copy_(embeddings["relEmbedding"])
        self.relHyper.weight.data.copy_(embeddings["relHyper"])
    def predictSimScore(self,head,relation,simMeasure="dot"):
        simMeasure = simMeasure.lower()
        assert (simMeasure.lower() in {"dot", "cos", "l2", "l1"})
        simScore = []
        head = self.entEmbedding(head)
        relation = self.relEmbedding(relation)
        expTailMatrix = head+relation
        hyperMatrix = self.relHyper(relation)
        tailEmbedding = self.entEmbedding.weight.data
        for expM, hypM in zip(expTailMatrix, hyperMatrix):
            hyperEmbedding = tailEmbedding - hypM.unsqueeze(0) * torch.matmul(tailEmbedding, hypM.unsqueeze(1))
            if simMeasure == "dot":
                simScore.append(torch.squeeze(torch.matmul(hyperEmbedding, hypM.unsqueeze(1))))
            elif simMeasure == "l2":
                score = torch.norm(hyperEmbedding - expM.unsqueeze(0), p=2, dim=1, keepdim=False)
                simScore.append(score)
            else:
                print("ERROR : simMeasure %s is not supported!" % simMeasure)
                exit(1)
        simScore = torch.vstack(simScore)
        return simScore
    