import torch
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

def train_func(model, sub_train_, BATCH_SIZE = 64, collate_fn = None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn = collate_fn, num_workers=0)
    for i, (x, y) in enumerate(data): 
        optimizer.zero_grad() 
        print(i)
        
        output = model(x)
        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == y).sum()
      
    # Adjust the learning rate
    scheduler.step()

    return torch.true_divide(train_loss , len(sub_train_)), torch.true_divide(train_acc, len(sub_train_))

def evaluate_model(model, data, BATCH_SIZE = 64, collate_fn = None):
    criterion = torch.nn.CrossEntropyLoss()

    loss = 0
    acc = 0
    data = DataLoader(data, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)

    l_expected = []
    l_actual = []

    for i, (x, y) in enumerate(data):
    
        with torch.no_grad():
            output = model(x)
            loss = criterion(output, y)
            loss += loss.item()
            acc += (output.argmax(1) == y).sum()

            l_expected.append(y)
            l_actual.append(output.argmax(1))

    t_expected  = torch.cat(l_expected,0)
    t_actual = torch.cat(l_actual,0)
    conf_matrix = confusion_matrix(t_expected,t_actual)
    plot_confusion_matrix(conf_matrix,['sport', 'entertainment', 'furniture', 'real estate', 'education', 'fashion', 'current affairs', 'game', 'technology', 'finance'])
    print(confusion_matrix)
    return torch.true_divide(loss, len(data)), torch.true_divide(acc, len(data)), conf_matrix




def plot_confusion_matrix(matrix,labels):
    normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(normalized,labels,labels)
    sn.set(font_scale=0.15)
    sn.heatmap(df_cm,annot=True,annot_kws={'size':8})
    plt.show()