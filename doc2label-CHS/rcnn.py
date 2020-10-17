import torch.nn as nn
import torch.nn.functional as F
import torch


class RCNN(nn.Module):
    def __RCNN__(self, output_size, hidden_size, vocab_size, emb_size, emb, dropout = 0.8):
        
        """
        output_size: number of classes, in this task, 10
        hidden_size: dim of rnn hidden layer
        vocab_size: number of word in word embedding
        emb_size: dim of word embedding
        emb: the weight of word embeddiing
        """
        super(RCNN, self).__init__()
		
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.emb.weight.data.copy_(torch.from_numpy(emb))

        self.lstm = nn.LSTM(emb_size, hidden_size, dropout = dropout, bidirectional=True, batch_first=True)

        self.w2 = nn.Linear(2*hidden_size+emb_size, hidden_size)
        self.label = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input = self.emb(input)


        output, (final_hidden, final_cell) = self.lstm(input)

        #concatnate the input embedding with the context
        output = torch.cat([input, output], 2)

        y = self.w2(output)

        y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
		y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
		y = y.squeeze(2)
		logits = self.label(y)
		
		return logits

