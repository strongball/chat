import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class EncoderRNN(nn.Module):
    def __init__(self, input_size, em_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.em_size = em_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, self.em_size)
        self.gru = nn.GRU(self.em_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input):
        output = self.embedding(input.long()).view(1, -1, self.em_size)
        output, hidden = self.gru(output)
        output = self.out(output[:,-1,:])
        return output, hidden
    
    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))

class DecoderRNN(nn.Module):
    def __init__(self, input_size, em_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.em_size = em_size

        self.embedding = nn.Embedding(input_size, em_size)
        self.gru = nn.GRU(em_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input.long()).view(1, -1, self.em_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output[0], hidden

    def initHidden(self):
        return Variable(torch.zeros(1, 1, self.hidden_size))
