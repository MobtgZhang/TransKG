import shutil
import json
import os
import pandas as pd
from collections import Counter
import logging
logger = logging.getLogger(__name__)

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
    shutil.move(rawpath, savepath)
def generateDict(dataPath,dictSaveDir):
    if type(dataPath)==str:
        logger.info("Loading standard data!")
        rawDict = readFile(dataPath)
    elif type(dataPath)==list:
        logger.info("Loading a list of standard data!")
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
    logger.info("Transform to index dict")
    entityDict = dict([(word, ind) for ind, word in enumerate(entityList)])
    relaDict = dict([(word, ind) for ind, word in enumerate(relaList)])

    # Save path
    entityDictPath = os.path.join(dictSaveDir, "entity_dict.json")
    relaDictPath = os.path.join(dictSaveDir, "relation_dict.json")

    # Save dicts
    json.dump({"stoi": entityDict, "itos": entityList}, open(entityDictPath, "w"))
    json.dump({"stoi": relaDict, 'itos': relaList}, open(relaDictPath, "w"))


