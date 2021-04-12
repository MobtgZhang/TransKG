import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
class tripleDataset(Dataset):
    def __init__(self,entityDictPath,relationDictPath,posDataPath):
        super(tripleDataset, self).__init__()
        print("INFO : Load entity and relation dict.")
        self.entityDict = json.load(open(entityDictPath, "r"))["stoi"]
        self.relationDict = json.load(open(relationDictPath, "r"))["stoi"]

        # Transform entity and relation to index
        print("INFO : Loading positive triples and transform to index.")
        self.posDf = pd.read_csv(posDataPath,
                                 sep="\t",
                                 names=["head", "relation", "tail"],
                                 header=None,
                                 encoding="utf-8",
                                 keep_default_na=False)
        self.transformToIndex(self.posDf, repDict={"head": self.entityDict,
                                                   "relation": self.relationDict,
                                                   "tail": self.entityDict})
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
