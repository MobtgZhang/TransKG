from config import Config
from transkg.process import generateDict,changeToStandard
from transkg.utils import printArgs
from transkg.dataloader import Trainer,Tester
from tensorboardX import SummaryWriter

def main():
    args = Config()
    printArgs(args)
    # change to standard dataset
    changeToStandard(args.train_raw,args.train_dir)
    changeToStandard(args.valid_raw,args.valid_dir)
    changeToStandard(args.test_raw,args.test_dir)
    # generate dictionary
    generateDict(dataPath=[args.train_dir,args.valid_dir,args.test_dir],
                 dictSaveDir=args.dict_dir)
    sumWriter = SummaryWriter(log_dir=args.summary_dir)
    trainModel = Trainer(args)
    trainModel.prepareData()
    trainModel.prepareModel()
    if args.load_embed:
        trainModel.loadPretrainEmbedding()
    if args.load_model:
        trainModel.loadPretrainModel()
    trainModel.run()
    trainModel.save()
    # test model
    testModel = Tester()
    testModel.test()
    sumWriter.close()
if __name__ == '__main__':
    # Print args
    main()