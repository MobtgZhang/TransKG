import os
import time
import logging
logger = logging.getLogger()

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from transkg import create_ents_rels,create_pos_neg_ids
from transkg import Dictionary,EntityrelationDataset
from transkg import TransE

def main(args):
    # generate the entities and relations
    data_dir = os.path.join(args.data_dir,args.dataset)
    result_dir = os.path.join(args.result_dir,args.dataset)
    # saved entity and relations files
    load_ent_file = os.path.join(result_dir,"entity2id.txt")
    load_rel_file = os.path.join(result_dir,"relation2id.txt")
    # raw train ,valid and test dataset
    tags_list = ["train","valid","test"]
    if not os.path.exists(load_ent_file) or not os.path.exists(load_rel_file):
        files_list = [os.path.join(data_dir,"%s.txt"%tag) for tag in tags_list]
        create_ents_rels(files_list,load_ent_file,load_rel_file)
        logger.info("The entity and relation saved in directory %s."%result_dir)
    else:
        logger.info("The files has already saved in directory %s"%result_dir)
    # load dictionary
    ent_dict= Dictionary.load(load_ent_file)
    rel_dict= Dictionary.load(load_rel_file)
    for tag in tags_list:
        save_filename = os.path.join(result_dir,"%s_ids.txt"%tag)
        if not os.path.exists(save_filename):
            load_filename = os.path.join(data_dir,"%s.txt"%tag)
            create_pos_neg_ids(load_filename,result_dir,ent_dict,rel_dict,tag,True if tag=="train" else False)
            logger.info("The entity and relation saved in file %s."%save_filename)
        else:
            logger.info("The files has already saved in file %s"%save_filename)
    
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    train_file = os.path.join(result_dir,"train_ids.txt")
    train_neg_file = os.path.join(result_dir,"train_neg_ids.txt")
    train_dataset = EntityrelationDataset(train_file,train_neg_file)
    train_loader = DataLoader(train_dataset,batch_size= args.batch_size,shuffle=True)
    valid_file = os.path.join(result_dir,"valid_ids.txt")
    valid_dataset = EntityrelationDataset(valid_file)
    valid_loader = DataLoader(valid_dataset,batch_size= args.batch_size,shuffle=False)
    test_file = os.path.join(result_dir,"test_ids.txt")
    test_dataset = EntityrelationDataset(test_file)
    test_loader = DataLoader(test_dataset,batch_size= args.batch_size,shuffle=False)
    
    # create models
    ent_size = len(ent_dict)
    rel_size = len(rel_dict)
    model = TransE(ent_size,rel_size,args.emb_dim)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=args.learning_rate,)
    for epoch in range(args.epoches):
        avg_loss = 0.0
        for item in train_loader:
            optimizer.zero_grad()
            pos_x,neg_x = item
            pos_x = pos_x.to(device)
            neg_x = neg_x.to(device)
            loss = model(pos_x,neg_x)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss /= len(train_loader)
        print(avg_loss)
        # test the dataset 
if __name__ == "__main__":
    # get and check argumentation
    args = config.get_args()
    config.check_args(args)
    # First step, create a logger
    logger = logging.getLogger()
    # The log level switch
    logger.setLevel(logging.INFO)
    # Second step, create a handler,which is used to write to logging file.
    args.time_step_str = time.strftime("%Y%m%d%H%M",time.localtime(time.time()))
    log_file = os.path.join(args.log_dir,args.time_step_str+".log")
    fh = logging.FileHandler(log_file, mode='w')
    # The log's switch of the output log file
    fh.setLevel(logging.DEBUG)
    # Third, define the output formatter
    format_str = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(format_str)
    ch = logging.StreamHandler()
    ch.setFormatter(format_str)
    # Fourth, add the logger into handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    main(args)




