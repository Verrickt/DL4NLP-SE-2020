import logging
import os.path
import jieba
import zhconv
import re
from gensim.corpora import WikiCorpus



class preprocessor():
    def __init__(self,src,dest):
        self.src = src
        self.dest= dest
        # regex used to filter out non-chinese characters
        self.regex =  re.compile('^[\u4e00-\u9fa5]+$')
        self.logging_interval = 10000
    def __bz2_to_sequence(self,input):
        # This function converts the compressed wikipedia corpus to a generator of articles
        logger = logging.getLogger("bz2_to_sequence")
        logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
        logging.root.setLevel(level=logging.INFO)
        logger.info("Reading corpus from %s" % (input))
        wiki = WikiCorpus(input,lemmatize=False,dictionary={})
        i = 0
        separator = " "
        for text in wiki.get_texts():
            yield separator.join(text)+"\n"
            i = i+1
            if i%self.logging_interval==0:
                logger.info("Extracted %d articles" % (i,))
        logger.info('Extraction finished')
    def __tokenize(self,str):
        return jieba.cut(str)
    def __cht_to_chs(self,input):
         chs = zhconv.convert(input,locale='zh-cn')
         return chs
    def __is_chs(self,input):
        return self.regex.search(input) is not None
    def preprocess(self):
        # This function perform the actual preprocessing
        if os.path.exists(self.dest):
            return
        with open(self.dest,'w+',encoding='utf-8') as o:
            i = 0
            for article in self.__bz2_to_sequence(self.src):
                # We're processing in a lazy-loading manner to save memory.
                chs = self.__cht_to_chs(article)
                tokens = self.__tokenize(chs)
                # Filter out non-chinese characters
                o.write(' '.join(filter(self.__is_chs,tokens)))
                o.write('\n')
                if(i%self.logging_interval==0):
                    o.flush()
                i = i + 1