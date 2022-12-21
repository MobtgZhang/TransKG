import torch
import torch.nn as nn
import torch.nn.functional as F

class TransD(nn.Module):
    def __init__(self,ent_size,rel_size,rel_dim,ent_dim,margin=1.0,l_regular=2):
        super(TransD,self).__init__()
        self.ent_size = ent_size
        self.ent_dim = ent_dim
        self.rel_size = rel_size
        self.rel_dim = rel_dim
        self.margin = margin
        self.l_regular = l_regular

        self.ent_emb = nn.Embedding(ent_size,ent_dim)
        self.ent_map_emb = nn.Embedding(ent_size,ent_dim)
        self.rel_emb = nn.Embedding(rel_size,rel_dim)
        self.rel_map_emb = nn.Embedding(rel_size,rel_dim)

        self.distfn = nn.PairwiseDistance(l_regular)
    def score_op(self,in_triple):
        head, relation, tail = torch.chunk(in_triple,chunks=3,dim=1)
        headp = torch.squeeze(self.ent_map_emb(head), dim=1)   # (B, 1, En) -> (B, En)
        head = torch.squeeze(self.ent_emb(head), dim=1)       # (B, 1, En) -> (B, En)
        tailp = torch.squeeze(self.ent_map_emb(tail), dim=1)   # (B, 1, En) -> (B, En)
        tail = torch.squeeze(self.ent_emb(tail), dim=1)       # (B, 1, En)  -> (B, En)
        relationp = torch.squeeze(self.rel_map_emb(relation), dim=1) # (B, 1, Em) -> (B, Em)
        relation = torch.squeeze(self.rel_emb(relation), dim=1)     # (B, 1, Em) -> (B, Em)

        relationp = torch.unsqueeze(relationp, dim=2)   # (B, Em, 1)
        headp = torch.unsqueeze(headp, dim=1)           # (B, 1, En)
        tailp = torch.unsqueeze(tailp, dim=1)           # (B, 1, En)

        I_mat = torch.eye(self.rel_dim,self.ent_dim).to(in_triple.device)
        Mrh = torch.matmul(relationp,headp) + I_mat
        Mrt = torch.matmul(relationp,tailp) + I_mat

        head = torch.unsqueeze(head, dim=2)
        tail = torch.unsqueeze(tail, dim=2)
        head = torch.squeeze(torch.matmul(Mrh, head), dim=2)   # (B, Em, 1) -> (B, Em)
        tail = torch.squeeze(torch.matmul(Mrt, tail), dim=2)   # (B, Em, 1) -> (B, Em)
        out = self.distfn(head+relation, tail)
        return out
    def forward(self,pos_x,neg_x):
        size = pos_x.shape[0]
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        return torch.sum(F.relu(pos_score-neg_score+self.margin))/size
    def tail_predict(self,in_triple,num_k):
        pass
    