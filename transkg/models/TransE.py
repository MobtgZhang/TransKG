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
    def tail_predict(self,in_triple,num_k):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)
        head_emb = torch.squeeze(self.ent_emb(head), dim=1)
        rel_emb = torch.squeeze(self.rel_emb(relation), dim=1)
        # transE model
        hr_emb = head_emb+rel_emb
        hr_emb = torch.unsqueeze(hr_emb, dim=1)
        hr_emb = hr_emb.expand(hr_emb.shape[0], self.ent_size, self.embedding_dim)
        t_emb = self.ent_emb.weight.data.expand(hr_emb.shape[0], self.ent_size, self.embedding_dim)
        # compute similarity: [batch_size, N]
        similarity = torch.norm(hr_emb - t_emb, dim=2)
        # indices: [batch_size, k]
        values, indices = torch.topk(similarity,num_k, dim=1, largest=False)
        # mean_indices: [batch_size, N]
        mean_values, mean_indices = torch.topk(similarity, self.ent_size, dim=1, largest=False)
        # tail: [batch_size] => [batch_size, 1]
        tail = tail.view(-1, 1)
        # result of hits10
        hits10 = torch.sum(torch.eq(indices, tail)).item()
        # result of mean rank
        mean_rank = torch.sum(torch.eq(mean_indices, tail).nonzero(), dim=0)[1]
        return hits10, mean_rank
    def head_predict(self,in_triple,num_k):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)
        tail_emb = torch.squeeze(self.ent_emb(tail), dim=1)
        rel_emb = torch.squeeze(self.rel_emb(relation), dim=1)
        # transE model
        rt_emb = tail_emb-rel_emb
        rt_emb = torch.unsqueeze(rt_emb, dim=1)
        rt_emb = rt_emb.expand(rt_emb.shape[0], self.ent_size, self.embedding_dim)
        h_emb = self.ent_emb.weight.data.expand(rt_emb.shape[0], self.ent_size, self.embedding_dim)
        # compute similarity: [batch_size, N]
        similarity = torch.norm(h_emb - rt_emb, dim=2)
        # indices: [batch_size, k]
        values, indices = torch.topk(similarity,num_k, dim=1, largest=False)
        # mean_indices: [batch_size, N]
        mean_values, mean_indices = torch.topk(similarity, self.ent_size, dim=1, largest=False)
        # head: [batch_size] => [batch_size, 1]
        head = head.view(-1, 1)
        # result of hits10
        hits10 = torch.sum(torch.eq(head,indices)).item()
        # result of mean rank
        mean_rank = torch.sum(torch.eq(head,mean_indices).nonzero(), dim=0)[1]
        return hits10, mean_rank
