#Uncomment the following line if running on huawei cloud
#import moxing as mox
import os
import preprocessor
import argparse
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

def training(corpus,target_path):
    # Model exists?
    if(os.path.exists(target_path+'/model')):
        return Word2Vec.load(target_path+'/model')
    model = Word2Vec(LineSentence(corpus), size=400, window=5, min_count=5,
            workers=multiprocessing.cpu_count())
    model.save(target_path+'/model')
    model.save_word2vec_format(target_path+'/vectors')
    return model

def eval(model):
    print("Evaluating model. Input a word to see the top3 most similar ones. Press Q to exit")
    while True:
        try:
            s = input()  
            if(s.lower()=='q'):
                 break;
            r = model.wv.most_similar(s,topn=3)
            print(r)
        except Exception as e:
            print(e)
            

    


def main(args):
    p = preprocessor.preprocessor(args.src, args.corpus_dest)
    p.preprocess()
    model = training(args.corpus_dest,args.model_path)
    eval(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a word2vec model')
    parser.add_argument('-src',type=str,help='Path to the source file')
    parser.add_argument('-corpus_dest',type=str,help='Path to save the processed corpus')
    parser.add_argument('-model_path',type=str,help='Path to the model')
    try:
        args = parser.parse_args()
        print("================================")
        print(args)
        print("================================")
        main(args)
    except:
        parser.print_help()
        sys.exit(0)

