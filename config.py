import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="./datasets",type=str)
    parser.add_argument("--result-dir",default="./result",type=str)
    parser.add_argument("--log-dir",default="./log",type=str)
    parser.add_argument("--dataset",default="fb15k",type=str)
    parser.add_argument("--batch-size",default=1024,type=int)
    parser.add_argument("--emb-dim",default=200,type=int)
    parser.add_argument("--learning-rate",default=0.01,type=float)
    parser.add_argument("--epoches",default=20,type=int)
    parser.add_argument("--num-k",default=10,type=int)
    parser.add_argument("--cuda",action="store_false")
    args = parser.parse_args()
    return args
def check_args(args):
    dataset_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    log_dir = os.path.join(args.log_dir,args.dataset)
    assert os.path.exists(dataset_dir)
    assert args.dataset in ["fb15k","fb15k-237","kg20c","wn18","wn18rr","yago3-10"]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    
