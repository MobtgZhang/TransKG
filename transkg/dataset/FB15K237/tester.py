import os
import logging
import json
logger = logging.getLogger()
from ..tester import Tester
from .data import prepareDataloader

class FB15K237Tester(Tester):
    def __init__(self,args):
        super(FB15K237Tester, self).__init__(args)
        self.ent_path = os.path.join(args.root_dir, args.dataset_name, "processed", "entity_dict.json")
        self.rel_path = os.path.join(args.root_dir, args.dataset_name, "processed", "relation_dict.json")
    def prepareData(self):
        logger.info("INFO : Prepare dataloader.")
        self.data_loader= prepareDataloader(self.root_dir,"test",self.eval_batch_size,self.shuffle,
                                              self.num_workers)
        self.entityDict = json.load(open(self.ent_path, mode="r"))
        self.relationDict = json.load(open(self.rel_path, mode="r"))
