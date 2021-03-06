import os
import logging
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

logger = logging.getLogger()
from ..model import TransE,TransH,TransR,TransD,TransA,KG2E
from ..model import NTN,LFM,SME
from ..utils import MREvaluation
class Trainer:
    def __init__(self,args):
        # Static arguments

        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.uuid_str = args.uuid_str
        self.sim_measure = args.sim_measure
        # Runtime arguments
        self.num_workers = args.num_workers
        self.num_epoches = args.num_epoches
        self.batch_size = args.batch_size
        self.save_steps = args.save_steps
        # Optimizer arguments
        self.opt_method = args.opt_method
        self.optimizer = None
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        # Data loader arguments
        self.shuffle = args.shuffle
        self.train_loader = None
        self.valid_loader = None
        self.entityDict = None
        self.relationDict = None
        # Model preparation arguments
        self.model_kargs = args.model_kargs
        self.model = None
        # File arguments
        self.root_dir = args.root_dir
        self.checkpoints_dir = args.checkpoints_dir
        self.emb_file = os.path.join(args.checkpoints_dir,args.model_name,args.model_name+"-"+self.uuid_str+"-"+"embeddings.npz")
        self.train_path = os.path.join(args.root_dir,args.dataset_name,"processed","train.txt")
        self.valid_path = os.path.join(args.root_dir,args.dataset_name,"processed","valid.txt")
        self.ent_path = os.path.join(args.root_dir,args.dataset_name,"processed","entity_dict.json")
        self.rel_path = os.path.join(args.root_dir,args.dataset_name,"processed","relation_dict.json")
        # Other model arguments
        self.use_gpu = args.use_gpu
    def prepareData(self, *args, **kwargs):
        raise NotImplementedError
    def prepareModel(self,kargs):
        logger.info("Init model %s" % self.model_name)
        ent_tot = len(self.entityDict["stoi"])
        rel_tot = len(self.relationDict["stoi"])
        if self.model_name == "TransE":
            self.model = TransE(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "TransA":
            self.model = TransA(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "TransH":
            self.model = TransH(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "TransD":
            self.model = TransD(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "TransR":
            self.model = TransR(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "KG2E":
            self.model = KG2E(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "SME":
            self.model = SME(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "NTN":
            self.model = NTN(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        elif self.model_name == "LFM":
            self.model = LFM(ent_tot=ent_tot,rel_tot=rel_tot,**kargs)
        else:
            logger.info("ERROR : No model named %s" % (self.model_name))
            exit(1)
    def loadPretrainEmbedding(self,filename):
        logger.info("INFO : Loading pre-training entity and relation embedding for model %s: %s!"%(self.model_name,filename))
        self.model.initialWeight(filename)
    def loadPretrainModel(self,filename):
        logger.info("INFO : Loading pre-training model:%s!" % filename)
        model_type = os.path.splitext(filename)[-1]
        if model_type == ".json":
            self.model.load_parameters(filename)
        elif model_type == ".ckpt":
            self.model.load_checkpoint(filename)
        elif model_type == ".mdl":
            self.model = torch.load(filename)
        else:
            logger.info("ERROR : Model type %s is not supported!" % filename)
            exit(1)
    def to_var(self,x,use_gpu):
        if use_gpu:
            return Variable(torch.LongTensor(x).cuda())
        else:
            return Variable(torch.LongTensor(x))
    def train_one_step(self,posX,negX):
        # normalize the embedding
        self.model.normalizeEmbedding()
        # calculate the loss score
        loss = self.model(self.to_var(posX,self.use_gpu),
                          self.to_var(negX,self.use_gpu))
        lossval = loss.cpu().item()
        # Calculate the gradient and step down
        # clear the gradients
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return lossval
    def run(self):
        if self.use_gpu:
            self.model.cuda()
        if self.opt_method.lower() == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.learning_rate,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method.lower() == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.opt_method.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        else:
            logger.info("Unknown optimizer method: %s"%self.opt_method)
            exit(1)
        logger.info("Finish initializing...")
        training_range = tqdm(range(self.num_epoches))
        root = os.path.join(self.checkpoints_dir, self.model_name)
        if not os.path.exists(root):
            logger.info("INFO : making dirs %s" % root)
            os.makedirs(root)
        loss_list = []
        mr_list = []
        for epoch in training_range:
            res = 0.0
            for posX, negX in self.train_loader:
                loss = self.train_one_step(posX, negX)
                res += loss
            # print the details of the model.
            # validation
            score = MREvaluation(self.valid_loader,self.model,self.sim_measure,self.use_gpu)
            mr_list.append(score)
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            loss_list.append(res)

            if self.save_steps and self.checkpoints_dir and (epoch) % self.save_steps == 0:
                model_path = os.path.join(root, self.uuid_str+"-" + str(epoch) + ".ckpt")
                logger.info("Save the model: %s"%model_path)
                self.model.save_checkpoint(model_path)
        loss_path = os.path.join(root,self.uuid_str+"-"+"loss.txt")
        with open(loss_path,mode="w",encoding="utf-8") as wfp:
            for item in loss_list:
                wfp.write(str(item)+"\n")
        logger.info("Save the loss: %s" % loss_path)
        mr_path = os.path.join(root, self.uuid_str + "-" + "mr.txt")
        with open(mr_path, mode="w", encoding="utf-8") as wfp:
            for item in mr_list:
                wfp.write(str(item) + "\n")
        logger.info("Save the mr: %s" % mr_path)
    def save(self):
        '''The method saves the model embedding,model and model parameters
        :return:
        '''
        # save the embedding
        output = self.model.retEvalWeights()
        root = os.path.join(self.checkpoints_dir, self.model_name)
        np.savez(os.path.join(root, self.uuid_str + "-embeddings.npz"), **output)
        # save model

        self.model_ck_file = os.path.join(root,self.uuid_str+ ".ckpt")
        self.model.save_checkpoint(self.model_ck_file)
        logger.info("Saved the model checkpoint file %s" % self.model_ck_file)
        # save model parameters
        self.parameters_file = os.path.join(root,self.uuid_str+ ".json")
        self.model.save_parameters(self.parameters_file)
        logger.info("Saved the parameters file %s"%self.parameters_file)
        # save model parameters
        self.model_file = os.path.join(root, self.uuid_str + ".mdl")
        torch.save(self.model,self.model_file)
        logger.info("Saved the model file %s" % self.model_file)

