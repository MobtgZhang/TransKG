import codecs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
class TransE(Model):
    '''
    This is the TransE model implementation.
    paper title: Translating Embeddings for Modeling Multi-relational Data. Curran Associates Inc. 2013.
    paper author: Bordes A , Usunier N , Garcia-Duran A , et al.
    paper website: http://www.thespermwhale.com/jaseweston/papers/CR_paper_nips13.pdf
    '''
    def __init__(self,ent_tot,rel_tot,emb_dim,margin=1.0,L=2):
        super(TransE, self).__init__(ent_tot,rel_tot)
        assert (L==1 or L==2)
        self.name = "TransE"
        self.margin = margin
        self.L = L
        self.emb_dim =emb_dim
        self.entEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=emb_dim)
        self.relEmbedding = nn.Embedding(num_embeddings=ent_tot,
                                         embedding_dim=emb_dim)
        # initialize the embedding layer
        nn.init.xavier_uniform_(self.entEmbedding.weight.data)
        nn.init.xavier_uniform_(self.relEmbedding.weight.data)
        self.distfn = nn.PairwiseDistance(L)
    def scoreOp(self,inputTriple):
        '''
        This function used to calculate the score,steps as follows:
        ==> Step1: Split input as head,relation and tail index column.
        ==> Step2: Transform index tensor to embedding tensor.
        ==> Step3: Sum head,relation and tail tensors with weights (1,1,-1).
        ==> Step4: Calculate distance as final score.
        :param inputTriple: The calculated data.
        :return:
        '''
        # Step1
        # head : shape(batch_size,1)
        # relation : shape(batch_size,1)
        # tail : shape(batch_size,1)
        head,relation,tail = torch.chunk(input=inputTriple,chunks=3,dim=1)
        # Step2
        # head: shape(batch_size,1,embedding_dim)
        # relation: shape(batch_size,1,embedding_dim)
        # tail: shape(batch_size,1,embedding_dim)
        head = torch.squeeze(self.entEmbedding(head),dim=1)
        tail = torch.squeeze(self.entEmbedding(tail),dim=1)
        relation = torch.squeeze(self.relEmbedding(relation),dim=1)
        # Step3 and Step4
        # output: shape(batch_size,embedding_dim) ==> shape(batch,1)
        output = self.distfn(head+relation,tail)
        return output
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
        return torch.sum(F.relu(input=posScore-negScore+self.margin))/size

    def predict(self, inputTriples):
        return self.scoreOp(inputTriples)
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
    def retEvalWeights(self):
        return {"entEmbedding":self.entEmbedding.weight.detach().cpu().numpy(),
                "relEmbedding":self.relEmbedding.weight.detach().cpu().numpy()}
    def initialWeight(self,filename):
        embeddings = np.load(filename, allow_pickle=True)
        self.entEmbedding.weight.data.copy_(embeddings["entEmbedding"])
        self.relEmbedding.weight.data.copy_(embeddings["relEmbedding"])