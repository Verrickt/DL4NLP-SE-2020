import re
import torch
import jieba

import numpy as np
from torch.utils.data import Dataset, DataLoader


class CorpusDataset(Dataset):

    def __init__(self, x, y):
        
        self.n_samples = len(y)

        # here the first column is the class label, the rest are the features
        self.x_data = x # size [n_samples, n_features]
        self.y_data = y # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def padding(insts):
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([[0]*(max_len-len(inst)) + inst   for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)
    return batch_seq


def collate_fn(batch_data):
   # print(batch_data)
    x = [d[0] for d in batch_data]
    y = [d[1] for d in batch_data]
    
    x = padding(x)
    y = torch.LongTensor(y)

    return x, y

def preprocess(file_name, emb_dict, tar_dict):
    target = []
    corpus = []

    with open(file_name, encoding='utf-8') as f:
        for i, s in enumerate(f.readlines()):
            s = (' '.join(jieba.cut(s)))
            s = re.sub(r'[{}]+'.format("。，！“”‘’`:"),'',s)
            s = s.split()
            #print(s[0])
            target.append(tar_dict[s[0]])
            corpus.append([emb_dict[w] for w in s[1:] if w in emb_dict])
    
    return corpus, target
    
