import torch
from transkg.utils import checkPath
class Config():
    '''
    Data arguments:This class is used for training model dataset parameters.
    '''
    def __init__(self):
        self.pospath = "./data/train.txt"
        self.validpath = ".data/valid.txt"
        self.testpath = ".data/valid.txt"
        self.entpath = "./source/dict/entityDict.json"
        self.relpath = "./source/dict/relationDict.json"
        self.embedpath = "./source/embed/"
        self.logpath = "./source/log/"
        self.savetype = "pt"

        # Dataloader arguments
        self.batch_size = 1024
        self.shuffle = True
        self.num_workers = 0
        self.drop_last = False
        self.repproba = 0.5
        self.exproba = 0.5

        # Model and training general arguments
        self.TransE = {"EmbeddingDim": 100,
                       "Margin": 1.0,
                       "L": 2}
        self.TransH = {"EmbeddingDim": 100,
                       "Margin": 1.0,
                       "L": 2,
                       "C": 0.01,
                       "Eps": 0.001}
        self.TransD = {"EntityDim": 100,
                       "RelationDim": 100,
                       "Margin": 2.0,
                       "L": 2}
        self.TransA = {"EmbeddingDim": 100,
                       "Margin": 3.2,
                       "L": 2,
                       "Lamb": 0.01,
                       "C": 0.2}
        self.KG2E = {"EmbeddingDim": 100,
                     "Margin": 4.0,
                     "Sim": "EL",
                     "Vmin": 0.03,
                     "Vmax": 3.0}
        self.use_gpu = torch.cuda.is_available()
        self.gpu_num = torch.cuda.device_count()
        self.model_name = "KG2E"
        self.weightdecay = 0
        self.epochs = 5
        self.evalepoch = 1
        self.learningrate = 0.01
        self.lrdecay = 0.96
        self.lrdecayepoch = 5
        self.optimizer = "Adam"
        self.evalmethod = "MR"
        self.simmeasure = "L2"
        self.modelsave = "param"
        self.modelpath = "./source/model/"
        self.load_embed = False
        self.load_model = False
        self.entityfile = "./source/embed/entityEmbedding.txt"
        self.relationfile = "./source/embed/relationEmbedding.txt"
        self.premodel = "./source/model/TransE_ent128_rel128.param"
        # Other arguments
        self.summarydir = "./source/summary/KG2E_EL/"
        # check path
        self.checkPath()
        # The parameters in papper

    def usePaperConfig(self):
        # Paper best params
        if self.model_name == "TransE":
            self.embeddingdim = 50
            self.learningrate = 0.01
            self.margin = 1.0
            self.distance = 1
            self.simmeasure = "L1"
        elif self.model_name == "TransH":
            self.batchsize = 1200
            self.embeddingdim = 50
            self.learningrate = 0.005
            self.margin = 0.5
            self.C = 0.015625
        elif self.model_name == "TransD":
            self.batchsize = 4800
            self.entitydim = 100
            self.relationdim = 100
            self.margin = 2.0
        else:
            raise TypeError("Unknown model type:%s"%self.model_name)

    def checkPath(self):
        # Check files
        checkPath(self.pospath)
        checkPath(self.validpath)
        # Check dirs
        checkPath(self.modelpath,raise_error=False)
        checkPath(self.logpath,raise_error=False)
        checkPath(self.summarydir,raise_error=False)
        checkPath(self.embedpath,raise_error=False)
