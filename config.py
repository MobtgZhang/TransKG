import os
import logging
import uuid

from transkg.utils import checkPath

logger = logging.getLogger()
ROOT_DIR = './data'
CHECKPOINTS_DIR = './checkpoints'
def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')
def add_args(parser):
    parser.register('type', 'bool', str2bool)

    # Model architecture
    model = parser.add_argument_group('Model parameters')
    model.add_argument('--model-name', type=str, default='TransE',
                       help='Model architecture type: TransE,TransH,TransR,TransD,TransA,KG2E,NTN,LFM,SME')
    model.add_argument('--ent-dim', type=int, default=100,
                       help='Embedding size of entity.')
    model.add_argument('--rel-dim', type=int, default=100,
                       help='Embedding size of relation.')
    # dataset details
    dataset = parser.add_argument_group('Dataset parameters')
    dataset.add_argument('--dataset-name', type=str, default="FB15K237",
                       help='Training the dataset for the model.')
    dataset.add_argument('--shuffle', type='bool', default=True,
                         help='Training process whether shuffle the dataset.')
    # Optimization details
    optim = parser.add_argument_group('Model Optimization')
    optim.add_argument('--opt-method', type=str, default='sgd',
                       help='Optimizer: sgd, adam, adagrad,adadelta')
    optim.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for sgd, adadelta')
    optim.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')
    optim.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--lr-decay', type=float, default=0.1,
                       help='learning rate decay factor')
    optim.add_argument('--momentum', type=float, default=0.1,
                       help='Momentum factor')
    optim.add_argument('--rho', type=float, default=0.95,
                       help='Rho for adadelta')
    optim.add_argument('--eps', type=float, default=1e-6,
                       help='Eps for adadelta')
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--cuda', type='bool', default=True,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=0,
                         help='Run on a specific GPU')
    runtime.add_argument('--num-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epoches', type=int, default=10,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=16,
                         help='Batch size for training')
    runtime.add_argument('--eval-batch-size', type=int, default=16,
                         help='Batch size for validation/test.')
    runtime.add_argument('--save-steps', type=int, default=2,
                         help='Save steps for the training model.')
    # Files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--root-dir', type=str, default=ROOT_DIR,
                       help='Directory for dataset.')
    files.add_argument('--checkpoints-dir', type=str, default=CHECKPOINTS_DIR,
                       help='Checkpoints files.')
    files.add_argument('--summary-dir', type=str, default=None,
                       help='Summary directory.')
    files.add_argument('--log-file', type=str, default='',
                       help='Checkpoints files.')
    files.add_argument('--pre-model', type=str, default=None,
                       help='Pretraining model for the dataset training.')
    files.add_argument('--emb-file', type=str, default=None,
                       help='Embedding files.')
    # Saving + loading
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    # validation
    validation = parser.add_argument_group('Validation/Test')
    validation.add_argument('--sim-measure', type=str, default="dot",
                           help='Evaluation method for the MR,including dot,cos,l1,l2.')
    validation.add_argument('--eval-method', type=str, default="MR",
                            help='Evaluation method for the model,including MR and Hit10.')
def set_default(args):
    uuid_value = uuid.uuid1()
    uuid_str = uuid_value.hex
    args.uuid_str = uuid_str
    args.log_file = os.path.join(args.checkpoints_dir,args.model_name,args.model_name+"-"+args.uuid_str+".log")
    args.summary_dir = os.path.join(args.checkpoints_dir,args.model_name,"summary")
    # model defination
def check_args(args):
    assert (args.model_name in ["TransE","TransH","TransR","TransD","TransA","KG2E","NTN","LFM","SME"])
    assert (args.dataset_name in ["FB15K237"])
    TransE = {"emb_dim": 100,
              "margin": 1.0,
              "L": 2}
    TransH = {"emb_dim": 100,
              "margin": 1.0,
              "L": 2,
              "C": 0.01,
              "eps": 0.001}
    TransD = {"ent_dim": 100,
              "rel_dim": 100,
              "margin": 2.0,
              "L": 2}
    TransA = {"emb_dim": 100,
              "margin": 3.2,
              "L": 2,
              "lamb": 0.01,
              "C": 0.2}
    TransR = {"emb_dim": 100,
              "margin": 1.0,
              "L": 2, }
    KG2E = {"emb_dim": 100,
            "margin": 4.0,
            "sim": "EL",
            "vmin": 0.03,
            "vmax": 3.0}
    SME = {"ent_dim": 100,
           "rel_dim": 100,
           "L": 2,
           "ele_dot": True}
    NTN = {"ent_dim": 100,
           "rel_dim": 100,
           "bias_flag": True,
           "rel_flag": True,
           "margin": 1.0}
    LFM = {

    }
    if args.model_name == "TransE":
        args.model_kargs = TransE
    elif args.model_name == "TransH":
        args.model_kargs = TransH
    elif args.model_name == "TransR":
        args.model_kargs = TransR
    elif args.model_name == "TransD":
        args.model_kargs = TransD
    elif args.model_name == "TransA":
        args.model_kargs = TransA
    elif args.model_name == "KG2E":
        args.model_kargs = KG2E
    elif args.model_name == "NTN":
        args.model_kargs = NTN
    elif args.model_name == "LFM":
        args.model_kargs = LFM
    elif args.model_name == "SME":
        args.model_kargs = SME
    else:
        logger.error("No model named %s" % (args.model_name))
        exit(1)
def check_paths(args):
    checkPath(args.root_dir,raise_error=False)
    checkPath(args.checkpoints_dir,raise_error=False)
    checkPath(os.path.join(args.checkpoints_dir, args.model_name),raise_error=False)
    checkPath(os.path.join(args.root_dir, args.dataset_name, "processed"), raise_error=False)
    checkPath(os.path.join(args.root_dir, args.dataset_name, "raw"), raise_error=False)
