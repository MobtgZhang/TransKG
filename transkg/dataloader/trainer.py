import os
import json
import torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .dataset import tripleDataset
from ..model import TransE,TransH,TransD,TransA,KG2E
def prepareTrainDataloader(train_path,entpath,relpath,
                          batch_size,shuffle,num_workers,drop_last):
    dataset = tripleDataset(posDataPath=train_path,
                            entityDictPath=entpath,
                            relationDictPath=relpath)
    dataset.generateNegSamples()
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader
def prepareValidDataloader(valid_path,entpath,relpath,
                          batch_size,shuffle,num_workers,drop_last):
    dataset = tripleDataset(posDataPath=valid_path,
                            entityDictPath=entpath,
                            relationDictPath=relpath)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader
class Trainer:
    def __init__(self,args):
        self.model_name = args.model_name
        self.work_threads = args.work_threads
        self.train_times = args.train_times
        self.batch_size = args.batch_size
        self.opt_method = args.opt_method
        self.optimizer = None
        self.lr_decay = args.lr_decay
        self.weight_decay = args.weight_decay
        self.alpha = args.alpha
        self.drop_last = args.drop_last
        self.shuffle = args.shuffle
        self.model_dict = args.model_dict
        self.pre_model = args.pre_model

        self.model = None
        self.train_loader = None
        self.valid_loader = None
        self.use_gpu = args.use_gpu
        self.save_steps = args.save_steps

        self.checkpoints_dir = args.checkpoints_dir
        self.train_dir = args.train_dir
        self.valid_dir = args.valid_dir
        self.ent_path = args.ent_path
        self.rel_path = args.rel_path
        self.entity_file = args.entity_file
        self.relation_file = args.relation_file
    def prepareData(self):
        print("INFO : Prepare dataloader.")
        self.train_loader = prepareTrainDataloader(self.train_dir,self.ent_path,self.rel_path,
                          self.batch_size,self.shuffle,self.work_threads,self.drop_last)
        #self.valid_loader = prepareValidDataloader(self.valid_dir, self.ent_path, self.rel_path,
        #                                           self.batch_size, self.shuffle, self.work_threads, self.drop_last)
        self.entityDict = json.load(open(self.ent_path,mode="r"))
        self.relationDict = json.load(open(self.rel_path,mode="r"))
    def prepareModel(self):
        print("INFO : Init model %s"%self.model_name)
        if self.model_name == "TransE":
            self.model = TransE(ent_tot=len(self.entityDict["stoi"]),
                                rel_tot=len(self.relationDict["stoi"]),
                                emb_dim=self.model_dict["TransE"]["EmbeddingDim"],
                                margin=self.model_dict["TransE"]["Margin"],
                                L=self.model_dict["TransE"]["L"])
        elif self.model_name == "TransA":
            self.model = None
        elif self.model_name == "TransH":
            self.model = None
        elif self.model_name == "TransD":
            self.model = None
        elif self.model_name == "TransR":
            self.model = None
        elif self.model_name == "KG2E":
            self.model = None
        else:
            print("ERROR : No model named %s"%(self.model_name))
            exit(1)
    def loadPretrainEmbedding(self):
        print("INFO : Loading pre-training entity and relation embedding:%s!"%self.model_name)
        if self.model_name == "TransE":
            self.model.initialWeight(entityEmbedFile=self.entity_file,
                                     entityDict=self.entityDict["stoi"],
                                     relationEmbedFile=self.relation_file,
                                     relationDict=self.relationDict["stoi"])
        elif self.model_name == "TransA":
            self.model = None
        elif self.model_name == "TransH":
            self.model = None
        elif self.model_name == "TransD":
            self.model = None
        elif self.model_name == "TransR":
            self.model = None
        elif self.model_name == "KG2E":
            self.model = None
        else:
            print("ERROR : No model named %s" % (self.model_name))
            exit(1)
    def loadPretrainModel(self):
        print("INFO : Loading pre-training model:%s!" % self.model_name)
        if self.model_name == "TransE":
            modelType = os.path.splitext(self.pre_model)[-1]
            if modelType == ".json":
                self.model.load_parameters(self.pre_model)
            elif modelType == ".pt":
                self.model.load_checkpoint(self.pre_model)
            else:
                print("ERROR : Model type %s is not supported!"%self.model_name)
                exit(1)
        elif self.model_name == "TransA":
            self.model = None
        elif self.model_name == "TransH":
            self.model = None
        elif self.model_name == "TransD":
            self.model = None
        elif self.model_name == "TransR":
            self.model = None
        elif self.model_name == "KG2E":
            self.model = None
        else:
            print("ERROR : No model named %s" % (self.model_name))
            exit(1)
    def run(self):
        if self.use_gpu:
            self.model.cuda()
        if self.opt_method.lower() == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")
        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for posX,negX in self.train_loader:
                loss = self.train_one_step(posX,negX)
                res += loss
            training_range.set_description("Epoch %d | loss: %f"%(epoch,res))
            if self.save_steps and self.checkpoints_dir and (epoch+1)%self.save_steps == 0:
                self.model.save_checkpoint(os.path.join(self.checkpoints_dir,self.model_name+"-"+str(epoch)+".ckpt"))
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
    def save(self):
        '''The method saves the model embedding,model and model parameters
        :return:
        '''
        # save the embedding
        output = self.model.retEvalWeights()
        entityEmbedding = output['entityEmbedding']
        relationEmbedding = output['relationEmbedding']
        np.savez(os.path.join(self.checkpoints_dir,"ent_embedding.npz"),entityEmbedding)
        np.savez(os.path.join(self.checkpoints_dir,"relationEmbedding.npz"),relationEmbedding)
        # save model
        self.model.save_checkpoint(os.path.join(self.checkpoints_dir, self.model_name + ".ckpt"))
        # save model parameters
        self.model.save_parameters(os.path.join(self.checkpoints_dir, self.model_name + ".json"))
    def to_var(self,x,use_gpu):
        if use_gpu:
            return Variable(torch.LongTensor(x).cuda())
        else:
            return Variable(torch.LongTensor(x))
    def set_model(self,model):
        self.model = model
    def set_opt_method(self,opt_method):
        self.opt_method = opt_method
    def set_train_times(self,train_times):
        self.train_times = train_times
    def set_save_steps(self,save_steps):
        self.save_steps = save_steps
    def set_checkpoints_dir(self,checkpoints_dir):
        self.checkpoints_dir = checkpoints_dir
    def set_lr_decay(self,lr_decay):
        self.lr_decay = lr_decay
    def set_weight_decay(self,weight_decay):
        self.weight_decay = weight_decay