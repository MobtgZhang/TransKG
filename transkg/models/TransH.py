import torch
import torch.nn as nn
import torch.nn,functional as F

class TransH(nn.Moudle):
    def __init__(self,ent_size,rel_size,emb_dim,margin=1.0,l_regular=2,v_c=1.0,eps=0.001):
        super(TransH,self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.margin = margin
        self.l_regular = l_regular
        self.v_c = v_c
        self.eps = eps

        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.rel_hyper = nn.Embedding(rel_size,emb_dim)
        self.distfn = nn.PairwiseDistance(l_regular)
    def forward(self,pos_x,neg_x):
        size = pos_x.shape[0]
        # Calculate score
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)
        # Get margin ranking loss
        # max(pos_score-neg_score+margin, 0)
        # Use F.relu()
        margin_loss = torch.sum(F.relu(input=pos_score-neg_score+self.margin))
        ent_loss = torch.sum(F.relu(torch.norm(self.ent_emb.weight, p=2, dim=1, keepdim=False)-1))
        orth_loss = torch.sum(F.relu(torch.sum(self.rel_hyper.weight * self.rel_emb.weight, dim=1, keepdim=False) / \
                                    torch.norm(self.rel_emb.weight, p=2, dim=1, keepdim=False) - self.eps ** 2))
        return  margin_loss/size +self.v_c*(ent_loss/self.ent_size + orth_loss/self.rel_size)
    def score_op(self,in_triple):
        # Step1
        head, relation, tail = torch.chunk(in_triple,chunks=3,dim=1)
        # Step2
        head_emb = torch.squeeze(self.ent_emb(head), dim=1)
        rel_hyper = torch.squeeze(self.rel_hyper(relation), dim=1)
        rel_emb = torch.squeeze(self.rel_emb(relation), dim=1)
        tail_emb = torch.squeeze(self.ent_emb(tail), dim=1)
        # Step3
        head_emb = head_emb - rel_hyper * torch.sum(head_emb * rel_hyper, dim=1, keepdim=True)
        tail_emb = tail_emb - rel_hyper * torch.sum(tail_emb * rel_hyper, dim=1, keepdim=True)
        # Step4
        return self.distfn(head_emb+rel_emb, tail_emb)
    def tail_predict(self,in_triple,num_k):
        pass


