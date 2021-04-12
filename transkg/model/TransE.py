import codecs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
class TransE(Model):
    '''
    This is the TransE model implementation.
    paper title:
    '''
    def __init__(self,ent_tot,rel_tot,emb_dim,margin=1.0,L=2):
        super(TransE, self).__init__(ent_tot,rel_tot)
        assert (L==1 or L==2)
        self.name = "TransE"
        self.margin = margin
        self.L = L
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
    def predict(self,data):
        pass
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
        return {"entityEmbedding":self.entEmbedding.weight.detach().cpu().numpy(),
                "relationEmbedding":self.relEmbedding.weight.detach().cpu().numpy()}
    def initialWeights(self,entityEmbedFile,entityDict,
                       relationEmbedFile,relationDict,fileType="txt"):
        '''
        Uesed to load pretraining entity and relation embedding.
        Implementation steps list as following:
        Method one: (Assign the pre-training vector one by one)
        ==> Step1: Read one line at a time,split the line as entity string and embed vector.
        ==> Step2: Transform the embed vector to np.array
        ==> Step3: Look up entityDict, find the index of the entity from entityDict, assign
                    the embed vector from step1 to the embedding matrix.
        ==> Step4: Repeat steps above until all line are checked.
        Method two: (Assign the pre-training at one time)
        ==> Step1: Initial a weight with the same shape of the embedding matrix.
        ==> Step2: Read every line of the EmbedFile and assign the vector to the initialized weight.
        ==> Step3: Assign the initialized weight to the embedding matrix at one time after all line are checked.
        :param entityEmbedFile:
        :param entityDict:
        :param relationEmbedFile:
        :param relationDict:
        :param fileType:
        :return:
        '''
        print("INFO : Loading entity pre-training embedding.")
        with codecs.open(entityEmbedFile,mode="r",encoding="utf-8") as rfp:
            _,embDim = rfp.readline().strip().split()
            assert (int(embDim) == self.entEmbedding.weight.size()[-1])
            for line in rfp:
                ent,embed = line.strip().split("\t")
                embed = np.array(embed.split(","),dtype=np.float32)
                if ent in entityDict:
                    self.entEmbedding.weight.data[entityDict[ent]].copy_(torch.from_numpy(embed))
        print("INFO : Loading relation pre-training embedding.")
        with codecs.open(entityEmbedFile,mode="r",encoding="utf-8") as rfp:
            _,embDim = rfp.readline().strip().split()
            assert (int(embDim) == self.relEmbedding.weight.size()[-1])
            for line in rfp:
                rel,embed = line.strip().split("\t")
                embed = np.array(embed.split(","),dtype=np.float32)
                if rel in entityDict:
                    self.relEmbedding.weight.data[relationDict[rel]].copy_(torch.from_numpy(embed))











