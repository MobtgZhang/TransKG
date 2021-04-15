import sys
import argparse
import logging

import numpy as np
import torch
from transkg.dataset.FB15K237 import FB15K237Dataset,FB15K237Trainer,preparingFB15273Dataset
from tensorboardX import SummaryWriter
from config import add_args,set_default,check_paths,check_args

logger = logging.getLogger(__name__)

def main():
    # preparing dataset
    preparingFB15273Dataset(args.root_dir)
    trainer = FB15K237Trainer(args)
    trainer.prepareData()
    trainer.prepareModel(args.model_kargs)
    if args.pre_model:
        trainer.loadPretrainModel(args.pre_model)
    if args.emb_file:
        trainer.loadPretrainEmbedding(args.emb_file)
    trainer.run()
    trainer.save()
if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'TransKG framework for training embeddings.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_args(parser)
    args = parser.parse_args()
    set_default(args)
    check_args(args)
    check_paths(args)
    # Set cuda
    args.use_gpu = args.cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    print(args)
    main()
