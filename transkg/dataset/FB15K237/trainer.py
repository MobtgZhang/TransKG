import logging
import json
logger = logging.getLogger(__name__)
from ..trainer import Trainer
from .data import prepareDataloader

class FB15K237Trainer(Trainer):
    def __init__(self,args):
        super(FB15K237Trainer, self).__init__(args)
    def prepareData(self):
        logger.info("INFO : Prepare dataloader.")
        self.train_loader = prepareDataloader(self.root_dir,"train",self.batch_size,self.shuffle,
                                              self.num_workers)
        self.valid_loader = prepareDataloader(self.root_dir,"valid",self.batch_size,self.shuffle,
                                              self.num_workers)
        self.entityDict = json.load(open(self.ent_path, mode="r"))
        self.relationDict = json.load(open(self.rel_path, mode="r"))
