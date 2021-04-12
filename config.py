import torch
from transkg.utils import checkPath
class Config():
    '''
    Data arguments:This class is used for training model dataset parameters.
    '''
    def __init__(self):
        self.train_dir = "./data/train.txt"
        self.valid_dir = "./data/valid.txt"
        self.test_dir = "./data/valid.txt"
        self.ent_path = "./source/dict/entityDict.json"
        self.rel_path = "./source/dict/relationDict.json"
        self.embed_path = "./source/embed/"
        self.checkpoints_dir = "./checkpoints"

        # Data loader arguments
        self.batch_size = 1024
        self.shuffle = True
        self.work_threads = 0
        self.drop_last = False
        self.repproba = 0.5
        self.exproba = 0.5
        self.save_steps = 25

        # Model and training general arguments
        TransE = {"EmbeddingDim": 100,
                       "Margin": 1.0,
                       "L": 2}
        TransH = {"EmbeddingDim": 100,
                       "Margin": 1.0,
                       "L": 2,
                       "C": 0.01,
                       "Eps": 0.001}
        TransD = {"EntityDim": 100,
                       "RelationDim": 100,
                       "Margin": 2.0,
                       "L": 2}
        TransA = {"EmbeddingDim": 100,
                       "Margin": 3.2,
                       "L": 2,
                       "Lamb": 0.01,
                       "C": 0.2}
        TransR = {

                                }
        KG2E = {"EmbeddingDim": 100,
                     "Margin": 4.0,
                     "Sim": "EL",
                     "Vmin": 0.03,
                     "Vmax": 3.0}
        self.model_dict = {"TransA":TransA,
                           "TransD":TransD,
                           "TransE":TransE,
                           "TransH":TransH,
                           "TransR":TransR,
                           "KG2E":KG2E}
        self.use_gpu = torch.cuda.is_available()
        self.gpu_num = torch.cuda.device_count()
        self.model_name = "TransE"
        self.alpha = 0
        self.weight_decay = 0
        self.train_times = 5
        self.evalepoch = 1
        self.learningrate = 0.01
        self.lr_decay = 0.96
        self.lrdecayepoch = 5
        self.opt_method = "Adam"
        self.evalmethod = "MR"
        self.simmeasure = "L2"
        self.modelsave = "param"
        self.modelpath = "./source/model/"
        self.load_embed = False
        self.load_model = False
        self.entity_file = "./source/embed/entityEmbedding.txt"
        self.relation_file = "./source/embed/relationEmbedding.txt"
        self.pre_model = None # "./source/model/TransE_ent128_rel128.param"
        # Other arguments
        self.summary_dir = "./source/summary/KG2E_EL/"
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
        checkPath(self.train_dir)
        checkPath(self.valid_dir)
        checkPath(self.test_dir)
        # Check dirs
        checkPath(self.modelpath,raise_error=False)
        checkPath(self.checkpoints_dir,raise_error=False)
        checkPath(self.summary_dir,raise_error=False)
        checkPath(self.embed_path,raise_error=False)
