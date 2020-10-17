import numpy as np
import chinese_datasets
import chinese_train
import rcnn 


if __name__ == "__main__":
    #处理中文词向量
    with open('sgns.renmin.word', 'r', encoding = 'utf-8') as f:
    
        tmp = list(map(int, f.readline().split()))
        vocab_size = tmp[0]
        emb_size = tmp[1]
        emb = np.zeros((vocab_size+1, emb_size))
        #convert word to id 
        emb_dict = {}
        emb_dict_cov = {}

        for i, line in enumerate(f.readlines()):
            line = line.split(' ')

            emb_dict[line[0]] = i+1
            emb_dict_cov[i+1] = line[0]
            emb[i+1, :] = list(map(np.float32, line[1:301]))
            if i % 1000 == 0:
                print(i/vocab_size*100, "%")

    #处理训练数据集
    tar_dict = {'体育': 0, '娱乐': 1, '家居':2, '房产':3, '教育':4, '时尚':5, '时政':6, '游戏':7, '科技':8, '财经': 9}

    train_c, train_t = chinese_datasets.preprocess('cnews.train.txt', emb_dict, tar_dict)
    
    #参数设置
    BATCH_SIZE = 64
    NUN_CLASS = 10
    HIDDEN_SIZE = 80
    EPOCH = 10
    #训练
    train_dataset = chinese_datasets.CorpusDataset(train_c, train_t)
    model = rcnn.RCNN(NUN_CLASS, HIDDEN_SIZE, emb.shape[0], emb.shape[1], emb)
    
    for i in range(EPOCH):
        chinese_train.train_func(model, train_dataset, collate_fn = chinese_datasets.collate_fn)
    
    #测试
    test_c, test_t = chinese_datasets.preprocess('cnews.test.txt', emb_dict, tar_dict)
    test_dataset = chinese_datasets.CorpusDataset(test_c, test_t)
    loss, acc, coff_matrix = chinese_train.evaluate_model(model, test_dataset, collate_fn = chinese_datasets.collate_fn)
