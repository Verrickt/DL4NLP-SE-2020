import torch
import random
from torchtext import data
from torchtext import datasets
import torch.nn.functional as F
import sys
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train(train_iter,valid_iter,model,args):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    steps = 0
    best_acc = 0
    last_step = 0
    for epoch in range(1,args.epochs+1):
        model.train()
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_()
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit,target)
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(valid_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)

def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_()
        if args.cuda:
                feature, target = feature.cuda(), target.cuda()
        logit = model(feature)
        loss = F.cross_entropy(logit, target, reduction='sum')

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy

def plot_confusion_matrix(matrix,labels):
    normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(normalized,labels,labels)
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm,annot=True,annot_kws={'size':16})
    plt.show()

def evaluate_model(model,data_iter,args):
     print('\nLoading model from {}...'.format(args.snapshot))
     model.load_state_dict(torch.load(args.snapshot))
     model.eval()
     corrects, avg_loss = 0, 0
     l_expected = []
     l_actual = []
     for batch in data_iter:
         feature, target = batch.text, batch.label
         feature.t_()
         if args.cuda:
                 feature, target = feature.cuda(), target.cuda()
         logit = model(feature)
         l_expected.append(target.to('cpu'))
         l_actual.append(logit.max(dim=1)[1].to('cpu'))
     t_expected  = torch.cat(l_expected,0)
     t_actual = torch.cat(l_actual,0)
     conf_matrix = confusion_matrix(t_expected,t_actual)
     plot_confusion_matrix(conf_matrix,['Positive','Negative'])
     print(confusion_matrix)
    
def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)