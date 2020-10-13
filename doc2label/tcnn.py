import torch
import torch.nn as nn
import torch.nn.functional as F

class Text_CNN(nn.Module):
    def __init__(self,args):
        super(Text_CNN,self).__init__()
        self.args = args
        I = args.embed_dim
        O = args.output_dim
        V = args.vocabulary_size
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V,I)
        self.convs = nn.ModuleList([nn.Conv2d(Ci,Co,kernel_size=(K,I)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(Ks)*Co,O)
        if self.args.static:
            self.embed.weight.requires_grad=False
    def forward(self,x):
        #(BatchSize,Width)
        x = self.embed(x) #(BatchSize,Width,I)
        x = x.unsqueeze(1) #(BatchSize,Ci,Width,I)
        x = [F.relu(conv(x).squeeze(3)) for conv in self.convs]  # [(BatchSize,Co,Width)]*len(Ks)
        x = [(F.max_pool1d(t,t.size(2))).squeeze(2) for t in x] # [(BatchSize,Co)]*len(Ks)
        x = torch.cat(x,1) # (BatchSize,Co*len(Ks))
        x = self.dropout(x)
        logit = self.fc(x)
        return logit
