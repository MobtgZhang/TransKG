import torch
import torch.nn as nn
import torch.nn.functional as F

class KG2E(nn.Module):
    def __init__(self, ent_size, rel_size, emb_dim, margin=1.0, sim="KL", vmin=0.03, vmax=3.0):
        super(KG2E, self).__init__()
        assert (sim in ["KL", "EL"])
        self.margin = margin
        self.sim = sim
        self.ent_size = ent_size
        self.rel_size = rel_size
        self.emb_dim = emb_dim
        self.vmin = vmin
        self.vmax = vmax

        # Embeddings represent the mean vector of entity and relation
        # Covars represent the covariance vector of entity and relation
        self.ent_emb = nn.Embedding(ent_size,emb_dim)
        self.ent_covar = nn.Embedding(ent_size,emb_dim)
        self.rel_emb = nn.Embedding(rel_size,emb_dim)
        self.rel_covar = nn.Embedding(rel_size,emb_dim)

    def KL_score(self,errorv,errorm,relationv,relationm):
        # Calculate KL(e, r)
        losep1 = torch.sum(errorv/relationv, dim=1)
        losep2 = torch.sum((relationm-errorm)**2 /relationv, dim=1)
        KLer = (losep1 + losep2 - self.emb_dim) / 2

        # Calculate KL(r, e)
        losep1 = torch.sum(relationv/errorv, dim=1)
        losep2 = torch.sum((errorm - relationm) ** 2 / errorv, dim=1)
        KLre = (losep1 + losep2 - self.emb_dim) / 2
        return (KLer + KLre) / 2
    def EL_score(self,errorv,errorm,relationv,relationm):
        losep1 = torch.sum((errorm - relationm) ** 2 / (errorv + relationv), dim=1)
        losep2 = torch.sum(torch.log(errorv+relationv), dim=1)
        return (losep1 + losep2) / 2
    def score_op(self,in_triple):
        head, relation, tail = torch.chunk(input=in_triple,chunks=3,dim=1)

        headm = torch.squeeze(self.ent_emb(head), dim=1)
        headv = torch.squeeze(self.ent_covar(head), dim=1)
        tailm = torch.squeeze(self.ent_emb(tail), dim=1)
        tailv = torch.squeeze(self.ent_covar(tail), dim=1)
        relationm = torch.squeeze(self.rel_emb(relation), dim=1)
        relationv = torch.squeeze(self.rel_covar(relation), dim=1)
        errorm = tailm - headm
        errorv = tailv + headv
        if self.sim == "KL":
            return self.KL_score(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        elif self.sim == "EL":
            return self.EL_score(relationm=relationm, relationv=relationv, errorm=errorm, errorv=errorv)
        else:
            print("ERROR : Sim %s is not supported!" % self.sim)
            exit(1)
    def forward(self,pos_x,neg_x):
        size = pos_x.shape[0]

        # Calculate score
        pos_score = self.score_op(pos_x)
        neg_score = self.score_op(neg_x)

        return torch.sum(F.relu(input=pos_score-neg_score+self.margin)) / size
    def tail_predict(self,in_triple,num_k):
        pass
