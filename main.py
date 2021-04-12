import os
from config import Config
from transkg.process import generateDict,changeToStandard
from transkg.utils import printArgs
from tensorboardX import SummaryWriter
from transkg.dataloader import Trainer
def main():
    # data preprocess
    trainRaw = "../data/freebase_mtr100_mte100-train.txt"
    validRaw = "../data/freebase_mtr100_mte100-valid.txt"
    testRaw = "../data/freebase_mtr100_mte100-test.txt"
    trainFile = "../data/train.txt"
    validFile = "../data/valid.txt"
    testFile = "../data/test.txt"
    dictDir = "../source/dict/"
    # change to standard dataset
    changeToStandard(trainRaw,trainFile)
    changeToStandard(validRaw,validFile)
    changeToStandard(testRaw,testFile)
    if not os.path.exists(dictDir):
        os.makedirs(dictDir)
    # generate dictionary
    generateDict(dataPath=[trainFile, validFile, testFile],
                 dictSaveDir=dictDir)
    # dataset preparing
    args = Config()
    printArgs(args)
    sumWriter = SummaryWriter(log_dir=args.summarydir)
    trainModel = Trainer(args)
    trainModel.prepareData()
    trainModel.prepareModel()
    if args.load_embed:
        trainModel.loadPretrainEmbedding()
    if args.load_model:
        pass
    sumWriter.close()
if __name__ == '__main__':
    # Print args
    main()