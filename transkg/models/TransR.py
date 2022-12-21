import torch
import torch.nn as nn
import torch.nn.functional as F

class TransR(nn.module):
    def __init__(self,ent_size,rel_size,emb_dim,margin=1.0,l_regular=2):
        super(TransR, self).__init__()
        assert l_regular in [1,2]
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.margin = margin
        self.l_regular = l_regular
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.transfer = nn.Parameter(torch.rand(size=(emb_dim,emb_dim)),requires_grad=True)
        self.distfn = nn.PairwiseDistance(l_regular)
    def score_op(self,in_triple):
        # Step1
        head,relation,tail = torch.chunk(input=in_triple,chunks=3,dim=1)
        # Step2
        head_m = torch.squeeze(self.ent_emb(head),dim=1)
        rel_m = torch.squeeze(self.rel_emb(relation),dim=1)
        tail_m = torch.squeeze(self.ent_emb(tail),dim=1)
        head_m = torch.matmul(head_m,self.transfer)
        tail_m = torch.matmul(tail_m,self.transfer)
        # Step3 and Step4
        score = self.distfn(head_m+rel_m,tail_m)
        return score
    def forward(self,pos_x,neg_x):
        size = pos_x.shape[0]
        # Calculate the score
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)
        # Use the margin ranking loss: max(posScore-negScore+margin,0)
        return torch.sum(F.relu(input=pos_score-neg_score+self.margin))/size
    def tail_predict(self,in_triple,num_k):
        pass
