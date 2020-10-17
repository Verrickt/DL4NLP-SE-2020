# DL4NLP-SE-2020
 
This repository contains the source code of assignment of DL4NLP-SE-2020

## word2vec
Trains a word2vec model based on wikipedia corpus.
### Usage
python main.py [-h] [-src SRC] [-corpus_dest CORPUS_DEST]
               [-model_path MODEL_PATH]
- h:show help
- src: The path to wikidump
- corpus_dest: The path to save the processed corpus
- model_path: The path to save the model.
### Dependencies
#### Required
1. Python 3.7
1. zhconv
1. jieba
1. gensim
#### Optional
cython

## doc2label

### doc2label-ENG
This code trains a [TCNN][2] for sentence classification with IMDB dataset

#### Requirements

The word embeddings used are based on [Glove][1]. Place `glove.6B.300d.txt` to `./vector_cache` to speed up the downloading


#### Dependencies
1. Pytorch
1. Torchtext
1. Sklearn
1. Seaborn
#### Reference

[Convolutional Neural Networks for Sentence Classification][2]

### doc2label-CHS

This code trains a [RCNN][3] for sentence classification with [THUCNews][5] dataset using the [Chinese Word Vector][4]

#### Requirement

1. Put `sgns.renmin.word` from [Chinese Word Vector][4] to current directory.

1. Put `cnews.train.txt`,`cnews.test.txt` from [THUCNews][5] to current directory

#### Dependencies
1. Pytorch
1. Sklearn
1. Seaborn
1. Jieba

[1]:https://nlp.stanford.edu/projects/glove/
[2]:https://arxiv.org/abs/1408.5882
[3]:https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745
[4]:https://github.com/Embedding/Chinese-Word-Vectors
[5]:http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews