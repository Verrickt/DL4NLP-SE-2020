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
This code trains a [TCNN][2] for sentence classification

### Requirements

The word embeddings used are based on [Glove][1]. Place `glove.6B.300d.txt` to `./vector_cache` to speed up the downloading


### Dependencies
1. Pytorch
1. Torchtext
1. Sklearn
1. Seaborn

### Reference

[Convolutional Neural Networks for Sentence Classification][2]

[1]:https://nlp.stanford.edu/projects/glove/
[2]:https://arxiv.org/abs/1408.5882