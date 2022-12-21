import torch
import torch.nn as nn
import torch.nn,functional as F

class TransA(nn.Moudle):
    def __init__(self,ent_size,rel_size,emb_dim,margin=1.0,l_regular=2,v_lambda=0.01,v_c=0.2):
        super(TransA,self).__init__()
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.margin = margin
        self.v_lambda = 0.01
        self.l_regular = l_regular
        self.v_lambda = v_lambda
        self.v_c = v_c
        
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.rel_w = nn.Parameter(torch.zeros(size=(rel_size,emb_dim,emb_dim)),requires_grad=True)
        self.distfn = nn.PairwiseDistance(l_regular)
    def forward(self,pos_x,neg_x):
        size = pos_x.shape[0]
        self.calculate_wr(pos_x,neg_x)

        # Calculate score
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        # Calculate loss
        margin_loss = torch.sum(F.relu(input=pos_score-neg_score+self.margin))/size
        wr_loss = torch.norm(input=self.rel_w, p=self.l_regular)/size
        weight_loss = torch.norm(input=self.ent_emb.weight, p=2)/self.ent_size + torch.norm(input=self.rel_emb.weight, p=2)/self.rel_size 
        return margin_loss + self.v_lambda * wr_loss + self.v_c * weight_loss
    def calculate_wr(self,pos_x,neg_x):
        pos_head, pos_rel, pos_tail = torch.chunk(input=pos_x,chunks=3,dim=1)
        neg_head, neg_rel, neg_tail = torch.chunk(input=neg_x,chunks=3,dim=1)
        pos_headm, pos_relm, pos_tailm = self.ent_emb(pos_head), self.rel_emb(pos_rel),self.ent_emb(pos_tail)
        neg_headm, neg_relm, neg_tailm = self.ent_emb(neg_head),self.rel_emb(neg_rel),self.ent_emb(neg_tail)
        error_pos = torch.abs(pos_headm + pos_relm - pos_tailm)
        error_neg = torch.abs(neg_headm + neg_relm - neg_tailm)
        del pos_headm, pos_relm, pos_tailm, neg_headm, neg_relm, neg_tailm
        self.rel_w[pos_rel] += torch.sum(torch.matmul(error_neg.permute((0, 2, 1)), error_neg), dim=0) - \
                           torch.sum(torch.matmul(error_pos.permute((0, 2, 1)), error_pos), dim=0)
    def score_op(self,in_triple):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)        
        head_emb = torch.squeeze(self.ent_emb(head), dim=1) 
        rel_emb = torch.squeeze(self.rel_emb(relation), dim=1)
        tail_emb = torch.squeeze(self.ent_emb(tail), dim=1)

        rel_wr = self.rel_w[relation]
        # (B, E) -> (B, 1, E) * (B, E, E) * (B, E, 1) -> (B, 1, 1) -> (B, )
        error = torch.unsqueeze(torch.abs(head_emb+rel_emb-tail_emb), dim=1)
        error = torch.matmul(torch.matmul(error, torch.unsqueeze(rel_wr, dim=0)), error.permute((0, 2, 1)))
        return torch.squeeze(error)
    def tail_predict(self,in_triple,num_k):
        pass

