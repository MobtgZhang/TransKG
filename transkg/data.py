import numpy as np
import torch

def load_txt(load_filename,index=False):
    all_dataset = []
    with open(load_filename,mode="r",encoding="utf-8") as rfp:
        for line in rfp:
            words = line.strip().split("\t")
            if index:
                words = [int(item) for item in words]
            all_dataset.append(words)
    return np.array(all_dataset,dtype=np.int64)
class Dictionary:
    def __init__(self,unk_token= "UNK_VAL"):
        self.words2id = {unk_token:0}
        self.id2words = [unk_token]
        self.start_id = -1
    def add(self,word):
        if word not in self.words2id:
            self.words2id[word] = len(self.words2id)
            self.id2words.append(word)
    def __getitem__(self,key):
        if type(key) == str:
            return self.words2id.get(key,0)
        elif type(key) == int:
            return self.id2words[key]
        else:
            raise TypeError("The key type %s is unknown."%str(type(key)))
    def __next__(self):
        if self.start_id>=len(self.id2words)-1:
            self.start_id = -1
            raise StopIteration()
        else:
            self.start_id += 1
            return self.id2words[self.start_id]
    def __iter__(self):
        return self
    def __repr__(self):
        re_str = "Dictionary (%d)"%len(self.id2words)
        return re_str
    def __str__(self):
        re_str = "Dictionary (%d)"%len(self.id2words)
        return re_str
    def __len__(self):
        return len(self.id2words)
    @staticmethod
    def load(load_file):
        tmp_dict = Dictionary()
        words2id = {}
        id2words = []
        with open(load_file,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                key,idx = line.strip().split("\t")
                id2words.append(key)
                words2id[key] = int(idx)
        tmp_dict.id2words = id2words
        tmp_dict.words2id = words2id
        return tmp_dict
    def save(self,save_file):
        with open(save_file,mode="w",encoding="utf-8") as wfp:
            for idx,key in enumerate(self.id2words):
                write_line = "%s\t%d\n"%(key,idx)
                wfp.write(write_line)
class EntityrelationDataset(torch.utils.data.Dataset):
    def __init__(self,file_name,neg_file_name=None):
        super(EntityrelationDataset,self).__init__()
        self.pos_dataset = load_txt(file_name,True)
        if neg_file_name is not None:
            self.neg_dataset = load_txt(neg_file_name,True)
            assert len(self.pos_dataset) == len(self.neg_dataset)
        else:
            self.neg_dataset = None
    def __getitem__(self,idx):
        if self.neg_dataset is not None:
            return self.pos_dataset[idx],self.neg_dataset[idx]
        else:
            return self.pos_dataset[idx]
    def __len__(self):
        return len(self.pos_dataset)

