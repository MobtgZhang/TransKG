import os
import json
import torch
from torch.utils.data import DataLoader
from .dataset import tripleDataset
from ..model import TransE,TransH,TransD,TransA,KG2E
def prepareEvalDataloader(args):
    dataset = tripleDataset(posDataPath=args.pospath,
                            entityDictPath=args.entpath,
                            relationDictPath=args.relpath)
    dataset.generateNegSamples()
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_workers,
                            drop_last=args.drop_last)
    return dataloader
class Trainer:
    def __init__(self,args):
        self.args = args
    def prepareData(self):
        print("INFO:Prepare dataloader.")
        self.eval_loader = prepareEvalDataloader(self.args)
        self.entityDict = json.load(open(self.args.entpath,mode="r"))
        self.relationDict = json.load(open(self.args.relpath,mode="r"))
    def prepareModel(self):
        print("INFO : Init model %s"%self.args.model_name)
        if self.args.model_name == "TransE":
            self.model = TransE(ent_tot=len(self.entityDict["stoi"]),
                                rel_tot=len(self.relationDict["stoi"]),
                                emb_dim=self.args.TransE["EmbeddingDim"],
                                margin=self.args.TransE["Margin"],
                                L=self.args.TransE["L"])
        elif self.args.model_name == "TransA":
            self.model = None
        elif self.args.model_name == "TransH":
            self.model = None
        elif self.args.model_name == "TransD":
            self.model = None
        elif self.args.model_name == "TransR":
            self.model = None
        elif self.args.model_name == "KG2E":
            self.model = None
        else:
            print("ERROR : No model named %s"%(self.args.model_name))
            exit(1)
        if self.args.use_gpu:
            with torch.cuda.device(self.args.gpu_num):
                self.model.cuda()
    def loadPretrainEmbedding(self):
        print("INFO : Loading pre-training entity and relation embedding:%s!"%self.args.model_name)
        if self.args.model_name == "TransE":
            self.model.initialWeight(entityEmbedFile=self.args.entityfile,
                                     entityDict=self.entityDict["stoi"],
                                     relationEmbedFile=self.args.relationfile,
                                     relationDict=self.relationDict["stoi"])
        elif self.args.model_name == "TransA":
            self.model = None
        elif self.args.model_name == "TransH":
            self.model = None
        elif self.args.model_name == "TransD":
            self.model = None
        elif self.args.model_name == "TransR":
            self.model = None
        elif self.args.model_name == "KG2E":
            self.model = None
        else:
            print("ERROR : No model named %s" % (self.args.model_name))
            exit(1)
    def run(self):
        pass