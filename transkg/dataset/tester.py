import os
import logging
import json

import numpy as np
import torch

from ..model import TransE,TransH,TransR,TransD,TransA,KG2E,BaseModule
from ..utils import MREvaluation
logger = logging.getLogger()
class Tester:
    def __init__(self,args):
        self.model_name = args.model_name
        self.eval_batch_size = args.eval_batch_size
        self.sim_measure = args.sim_measure
        # Files for the test
        self.checkpoints_dir = args.checkpoints_dir
        self.root_dir = args.root_dir
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers
        self.use_gpu = args.use_gpu
        self.data_loader = None
        self.model = BaseModule()
    def load_embeddings(self,filename):
        self.embeddings = np.load(filename)
    def prepareDataloader(self):
        raise NotImplementedError
    def run_link_prediction(self):
        # inintialize test preparing
        if self.use_gpu:
            self.model.cuda()
        score = MREvaluation(self.data_loader,self.model,simMeasure=self.sim_measure,use_gpu=self.use_gpu)
        logger.info("MR score for test dataset is %f"%score)
    def set_model(self,filename):
        self.model = torch.load(filename)

