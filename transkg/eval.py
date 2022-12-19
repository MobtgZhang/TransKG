import pandas as pd
import torch

class DataSaver:
    def __init__(self,num_k,save_file):
        self.num_k = num_k
        self.save_file = save_file
        self.dataset = ["hits%d"%num_k,"mean-rank"]
    def add(self,tmpdict):
        tp_list = [tmpdict["hits%d"%self.num_k],tmpdict["mean-rank"]]
        self.dataset.append(tp_list)
        df_values = pd.DataFrame(self.dataset[1:],columns=self.dataset[0])
        df_values.to_csv(self.save_file,index=None)
def hits_value(model,data_loader,device,num_k=10):
    hitsN, mean_rank = 0, 0
    model.eval()
    for item in data_loader:
        in_triple = item.to(device)
        hN,mr = model.tail_predict(in_triple,num_k)
        hitsN+=hN
        mean_rank+=mr
    length = len(data_loader.dataset)
    hitsN /= length
    mean_rank /= length
    return {
        "hits%d"%num_k:hitsN,
        "mean-rank":mean_rank
    }

