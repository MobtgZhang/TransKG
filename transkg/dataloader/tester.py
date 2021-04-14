import os
import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .dataset import tripleDataset
from ..utils import MREvaluation

def prepareDataloader(dataset_path,entpath,relpath,
                          batch_size,shuffle,num_workers):
    dataset = tripleDataset(posDataPath=dataset_path,
                            entityDictPath=entpath,
                            relationDictPath=relpath)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
class Tester:
    def __init__(self,args):
        self.model_name = args.model_name
        self.work_threads = args.work_threads
        self.train_times = args.train_times
        self.batch_size = args.batch_size
        self.use_gpu = args.use_gpu
        self.checkpoints_dir = args.checkpoints_dir
        self.ent_path = args.ent_path
        self.rel_path = args.rel_path
        self.test_dir = args.test_dir
        self.model_dict = args.model_dict
        self.pre_model = args.pre_model
        self.sim_measure = args.sim_measure
        # dataset loader
        self.data_loader = None
    def load_embedding(self,filename):
        self.embeddings = np.load(filename, allow_pickle=True)
    def run_link_prediction(self):
        # initialize test preparing
        root = os.path.join(self.checkpoints_dir, self.model_name)
        if not os.path.exists(root):
            print("INFO : making dirs %s" % root)
            os.makedirs(root)
        MeanRanks = MREvaluation(self.data_loader,self.model_name,simMeasure=self.sim_measure,**self.embeddings)
        print("MR score for model: %s is %f."%(self.model_name,MeanRanks))
        with open(os.path.join(self.checkpoints_dir,self.model_name,"result.txt"),mode="w",encoding="utf-8") as wfp:
            wfp.write("mr score: %f"%MeanRanks)
    def load_data(self,filename):
        print("INFO : Prepare test dataloader.")
        self.data_loader = prepareDataloader(filename, self.ent_path, self.rel_path,
                                                   self.batch_size, False, self.work_threads)
        self.entityDict = json.load(open(self.ent_path, mode="r"))
        self.relationDict = json.load(open(self.rel_path, mode="r"))
    def set_model(self,model):
        self.model = model
    def set_use_gpu(self,use_gpu):
        self.use_gpu = use_gpu
    def to_var(self,x,use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))