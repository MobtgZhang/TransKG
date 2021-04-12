import os
import json
import shutil
from collections import Counter
def readFile(filename):
    head = {}
    relation = {}
    tail = {}
    with open(filename,mode="r",encoding="utf-8") as rfp:
        while True:
            line = rfp.readline()
            if not line:
                break
            outVecs = line.strip().split()
            head[outVecs[0]] = len(head)
            relation[outVecs[1]] = len(relation)
            tail[outVecs[2]] = len(tail)
    return {"head":head,"relation":relation,"tail":tail}
def changeToStandard(rawpath,savepath):
    shutil.copy(rawpath, savepath)
def generateDict(dataPath,dictSaveDir):
    if type(dataPath)==str:
        print("INFO : Loading standard data!")
        rawDict = readFile(dataPath)
    elif type(dataPath)==list:
        print("INFO : Loading a list of standard data!")
        rawDict = {"head":{},"relation":{},"tail":{},}
        for filename in dataPath:
            dictTmp = readFile(filename)
            rawDict["head"].update(dictTmp["head"])
            rawDict["relation"].update(dictTmp["relation"])
            rawDict["tail"].update(dictTmp["tail"])
    headCounter = Counter(rawDict["head"])
    relationCounter = Counter(rawDict["relation"])
    tailCounter = Counter(rawDict["tail"])

    # Generate entity and relation list
    entityList = list((headCounter+tailCounter).keys())
    relaList = list(relationCounter.keys())

    # Transform to index dict
    print("INFO : Transform to index dict")
    entityDict = dict([(word, ind) for ind, word in enumerate(entityList)])
    relaDict = dict([(word, ind) for ind, word in enumerate(relaList)])

    # Save path
    entityDictPath = os.path.join(dictSaveDir, "entityDict.json")
    relaDictPath = os.path.join(dictSaveDir, "relationDict.json")

    # Save dicts
    json.dump({"stoi": entityDict, "itos": entityList}, open(entityDictPath, "w"))
    json.dump({"stoi": relaDict, 'itos': relaList}, open(relaDictPath, "w"))
if __name__ == "__main__":
    trainFile = "../data/freebase_mtr100_mte100-train.txt"
    validFile = "../data/freebase_mtr100_mte100-valid.txt"
    testFile = "../data/freebase_mtr100_mte100-test.txt"
    saveDir = "../data/"
    dictDir = "../source/dict/"
    if not os.path.exists(dictDir):
        os.makedirs(dictDir)
    generateDict(dataPath=[trainFile,validFile,testFile],
                 dictSaveDir=dictDir)
