import torch
import torch.nn.functional as F
import torch.nn as nn

class TransE(nn.Module):
    def __init__(self,ent_size,rel_size,embedding_dim, margin=1.0, l_regular=2):
        super(TransE,self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.l_regular = l_regular
        self.ent_emb = nn.Embedding(ent_size,embedding_dim)
        self.rel_emb = nn.Embedding(rel_size,embedding_dim)
        self.distfn = nn.PairwiseDistance(l_regular)
    def score_op(self,in_triple):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)
        head = torch.squeeze(self.ent_emb(head), dim=1)
        tail = torch.squeeze(self.ent_emb(tail), dim=1)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)
        loss = self.distfn(head+relation,tail)
        return loss
    def forward(self,pos_x, neg_x):
        size = pos_x.shape[0]
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)
        return torch.sum(F.relu(pos_score-neg_score+self.margin))/size
