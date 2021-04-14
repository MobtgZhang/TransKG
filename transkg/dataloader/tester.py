import os
import torch
import json
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .dataset import tripleDataset

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
        # dataset loader
        self.data_loader = None
    def run_link_prediction(self):
        # initialize test preparing
        root = os.path.join(self.checkpoints_dir, self.model_name)
        if not os.path.exists(root):
            print("INFO : making dirs %s" % root)
            os.makedirs(root)
        for data in self.data_loader:
            pass

    def load_model(self,pre_model):
        modelType = os.path.splitext(pre_model)[-1]
        if modelType == ".json":
            self.model.load_parameters(pre_model)
        elif modelType == ".pt":
            self.model.load_checkpoint(pre_model)
        else:
            print("ERROR : Model type %s is not supported!" % self.model_name)
            exit(1)
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

