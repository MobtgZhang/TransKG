import shutil

import requests
import os
import tarfile
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import logging
logger = logging.getLogger(__name__)

from .utils import generateDict,changeToStandard

def download(raw_dir,download_url):
    filename = os.path.join(raw_dir, "fb15k.tgz")
    logger.info("Download the FB15K-237 dataset,raw file is in path: %s" % (filename))
    res = requests.get(download_url,stream=True)
    file_size = int(res.headers.get('Content-Length'))
    pbar = tqdm(total=file_size)
    tip_len = 1024
    with open(os.path.join(raw_dir, "fb15k.tgz"), 'wb') as wfp:
        for chunk in res.iter_content(tip_len):
            wfp.write(chunk)
            pbar.set_description('Download file: %s'%(filename))
            pbar.update(tip_len)  # 更新进度条长度
        pbar.close()
    logger.info("Extracte file: %s" % (filename))
    tar = tarfile.open(filename)
    tar.extractall(raw_dir)
    logger.info("Delete file: %s" % (filename))
    os.remove(filename)
    shutil.move(os.path.join(raw_dir,"FB15k","freebase_mtr100_mte100-test.txt"),os.path.join(raw_dir,"freebase_mtr100_mte100-test.txt"))
    shutil.move(os.path.join(raw_dir,"FB15k","freebase_mtr100_mte100-train.txt"),os.path.join(raw_dir,"freebase_mtr100_mte100-valid.txt"))
    shutil.move(os.path.join(raw_dir,"FB15k","freebase_mtr100_mte100-valid.txt"),os.path.join(raw_dir,"freebase_mtr100_mte100-train.txt"))
    shutil.move(os.path.join(raw_dir,"FB15k","README"),os.path.join(raw_dir,"README"))
    shutil.rmtree(os.path.join(raw_dir,"FB15k"))
    logger.info("The file save in path: %s" % (raw_dir))

def preparingFB15273Dataset(root_dir="./data"):
    root_dir = os.path.join(root_dir,"FB15K237")
    raw_dir = os.path.join(root_dir, "raw")
    processed_dir = os.path.join(root_dir, "processed")
    entity_path = os.path.join(processed_dir, "entity_dict.json")
    relation_path = os.path.join(processed_dir, "relation_dict.json")
    raw_train_path = os.path.join(raw_dir,"freebase_mtr100_mte100-train.txt")
    raw_valid_path = os.path.join(raw_dir, "freebase_mtr100_mte100-valid.txt")
    raw_test_path = os.path.join(raw_dir, "freebase_mtr100_mte100-test.txt")
    raw_readme_path = os.path.join(raw_dir, "README")
    os_flag = os.path.exists(raw_train_path) and \
              os.path.exists(raw_valid_path) and \
              os.path.exists(raw_test_path) and \
              os.path.exists(raw_readme_path)
    if not os_flag:
        logger.warning("Can't find the FB15K-237 dataset in path %s" % (raw_dir))
        try:
            os.remove(raw_test_path)
            os.remove(raw_valid_path)
            os.remove(raw_train_path)
            os.remove(raw_readme_path)
        except FileNotFoundError:
            pass
        download_url = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz"
        download(raw_dir,download_url)
    train_path = os.path.join(processed_dir, "train.txt")
    valid_path = os.path.join(processed_dir, "valid.txt")
    test_path = os.path.join(processed_dir, "test.txt")
    if not os.path.exists(train_path):
        logger.info("Change file %s to standard file." % (raw_train_path))
        changeToStandard(raw_train_path,train_path)
    if not os.path.exists(valid_path):
        logger.info("Change file %s to standard file." % (raw_valid_path))
        changeToStandard(raw_valid_path,valid_path)
    if not os.path.exists(test_path):
        logger.info("Change file %s to standard file." % (raw_test_path))
        changeToStandard(raw_test_path,test_path)
    if not (os.path.exists(entity_path) and os.path.exists(relation_path)):
        try:
            os.remove(entity_path)
            os.remove(relation_path)
        except FileNotFoundError:
            pass
        logger.info("Genrate entity and relation dictionary.")
        generateDict(dataPath=[train_path,valid_path,test_path],dictSaveDir=processed_dir)
def prepareDataloader(root_dir,dtype,batch_size,shuffle,num_workers):
    dataset = FB15K237Dataset(root_dir,dtype)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader
class FB15K237Dataset(Dataset):
    def __init__(self,root_dir="./data",dtype = "train"):
        # entityDictPath, relationDictPath, posDataPath
        super(FB15K237Dataset, self).__init__()
        assert (dtype in ["train","valid","test"])
        self.root_dir = root_dir
        self.processed_dir = os.path.join(root_dir,"FB15K237","processed")
        self.entity_path = os.path.join(self.processed_dir,"entity_dict.json")
        self.relation_path = os.path.join(self.processed_dir,"relation_dict.json")
        if dtype == "train":
            self.data_path = os.path.join(self.processed_dir,"train.txt")
        if dtype == "valid":
            self.data_path = os.path.join(self.processed_dir,"valid.txt")
        if dtype == "test":
            self.data_path = os.path.join(self.processed_dir,"test.txt")
        logger.info("Load entity and relation dict.")
        self.entityDict = json.load(open(self.entity_path, "r"))["stoi"]
        self.relationDict = json.load(open(self.relation_path, "r"))["stoi"]

        # Transform entity and relation to index
        logger.info("Loading positive triples and transform to index.")
        self.posDf = pd.read_csv(self.data_path,
                                 sep="\t",
                                 names=["head", "relation", "tail"],
                                 header=None,
                                 encoding="utf-8",
                                 keep_default_na=False)
        self.transformToIndex(self.posDf, repDict={"head": self.entityDict,
                                                   "relation": self.relationDict,
                                                   "tail": self.entityDict})
        if dtype == "train":
            self.generateNegSamples()
    def generateNegSamples(self,repProba=0.5,exProba=0.5,repSeed=0,exSeed=0,headSeed=0,tailSeed=0):
        assert (repProba>=0 and repProba<=1.0 and exProba>=0 and exProba<=1.0)
        # Generate negtive samples from positive samples
        print("INFO : Generate negtive samples from positive samples.")
        self.negDf = self.posDf.copy()
        np.random.seed(repSeed)
        repProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negDf),))
        np.random.seed(exSeed)
        exProbaDistribution = np.random.uniform(low=0.0, high=1.0, size=(len(self.negDf),))
        shuffleHead = self.negDf["head"].sample(frac=1.0, random_state=headSeed)
        shuffleTail = self.negDf["tail"].sample(frac=1.0, random_state=tailSeed)

        # The method to replace head or tail
        def replaceHead(relHead,shuffHead,shuffTail,repP,exP):
            if repP >= repProba:
                return  relHead
            else:
                if exP >exProba:
                    return shuffHead
                else:
                    return shuffTail
        def replaceTail(relTail,shuffHead,shuffTail,repP,exP):
            if repP <repProba:
                return relTail
            else:
                if exP >exProba:
                    return shuffTail
                else:
                    return shuffHead
        self.negDf["head"] = list(map(replaceHead,self.negDf["head"],shuffleHead,shuffleTail,
                                      repProbaDistribution,exProbaDistribution))
        self.negDf["tail"] = list(map(replaceTail,self.negDf["tail"],shuffleHead,shuffleTail,
                                      repProbaDistribution,exProbaDistribution))

    @staticmethod
    def transformToIndex(csvData:pd.DataFrame, repDict:dict):
        for col in repDict.keys():
            csvData[col] = csvData[col].apply(lambda x:repDict[col][x])
    def __len__(self):
        return len(self.posDf)
    def __getitem__(self, item):
        if hasattr(self,"negDf"):
            return np.array(self.posDf.iloc[item,:3]),np.array(self.negDf.iloc[item,:3])
        else:
            return np.array(self.posDf.iloc[item, :3])
