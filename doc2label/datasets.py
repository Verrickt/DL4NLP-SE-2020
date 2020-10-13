import torch
import random
from torchtext import data
from torchtext import datasets
import torch.nn.functional as F
def imdb(text_field,label_field,args):
    print('Loading data...')
    train_data, test_data = datasets.IMDB.splits(text_field,label_field)
    train_data, valid_data = train_data.split(random_state = random.seed(args.seed))
    print('Loading GloVe...')
    text_field.build_vocab(train_data,valid_data,vectors='glove.6B.300d'
                           ,max_size = args.max_vocabulary_size
                           ,unk_init=torch.Tensor.normal_,vectors_cache='./vector_cache')
    label_field.build_vocab(train_data,valid_data)
    train_iter,valid_iter,test_iter = data.Iterator.splits((train_data,valid_data,test_data),batch_size=args.batch_size)
    return train_iter,valid_iter,test_iter

