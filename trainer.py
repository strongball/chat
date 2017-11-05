from chatTool import *
from Net import *
import pickle
import random
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import optim

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epoch', help="Epoch to Train", type=int, default=10)
parser.add_argument('-lr', help="Loss to Train", type=float, default = 1e-4)
parser.add_argument('-m', help="Model dir", required=True)
parser.add_argument('-t', help="Teach forceing rate", type=float, default = 0.5)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

LangBag = "dict.pkl"
DataName = "./dgk_lost_conv/results/lost.conv.tconv"

EncoderFile = os.path.join(args.m, "Encoder.pth") #"Encoder.pth"
DecoderFile = os.path.join(args.m, "Decoder.pth")#"Decoder.pth"

with open(LangBag, 'rb') as f:
    lang = pickle.load(f)

trainset = Reader(DataName)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

INPUTSIZE = lang.n_words

if not os.path.isdir(args.m):
    os.mkdir(args.m)
if os.path.isfile(EncoderFile):
    encoder = torch.load(EncoderFile)
    print("Load encoder model: {}.".format(EncoderFile))
else:
    encoder = EncoderRNN(INPUTSIZE, 256, 1024)
    print("Create encoder model.")

if os.path.isfile(DecoderFile):
    decoder = torch.load(DecoderFile)
    print("Load decoder model: {}.".format(DecoderFile))
else:
    decoder = DecoderRNN(INPUTSIZE, 256, 1024, INPUTSIZE)
    print("Create decoder model.")

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
def main():
    print("Start training........\n")
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()
    tfrate = args.t
    for epoch in range(args.epoch):
        sum = 0
        for i,data in enumerate(trainloader, 0):
            input, target = data
            input = torch.Tensor(lang.sentenceToVector(input[0])).long()
            target = torch.Tensor(lang.sentenceToVector(target[0], eos=True)).long()

            if use_cuda:
                input = input.cuda()
                target = target.cuda()

            inputLen = input.size()[0]
            targetLen = target.size()[0]

            input = Variable(input)
            target = Variable(target)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()


            o, encoder_hidden = encoder(input)
            decoder_input = Variable(torch.LongTensor([0]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_hidden = o.view(1,1,-1)

            loss = 0
            use_teacher_forcing = True if random.random() < tfrate else False
            for di in range(targetLen):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0]
                decoder_input = target[di]
                if use_teacher_forcing:
                    tfrate *= 0.95
                    decoder_input = Variable(torch.LongTensor([ni]))
                    if use_cuda:
                        decoder_input = decoder_input.cuda()
                else:
                    decoder_input = target[di]

                loss += criterion(decoder_output, target[di])
            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            sum += loss.data[0] / targetLen
            if (i+1) % 100 == 0:
                print("Epoch: {}, Step {}, loss: {}".format(epoch, i, sum / 100))
                sum = 0
                predit(i)
            if (i+1) % 5000 == 0:
                torch.save(encoder, EncoderFile)
                torch.save(decoder, DecoderFile)
                print("Save model {}, {}".format(EncoderFile, DecoderFile))
        torch.save(encoder, EncoderFile)
        torch.save(decoder, DecoderFile)
        print("Save model {}, {}".format(EncoderFile, DecoderFile))

def predit(n):
    q,a = trainset[n]

    input = torch.Tensor(lang.sentenceToVector(q)).long()
    input = Variable(input)
    encoder_hidden = encoder.initHidden()
    if use_cuda:
        input = input.cuda()
        encoder_hidden = encoder_hidden.cuda()
    o, encoder_hidden = encoder(input)
    decoder_input = Variable(torch.LongTensor([0]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = o.view(1,1,-1)
    ans = []
    for i in range(50):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0]
        ans.append(ni)
        decoder_input = Variable(torch.LongTensor([ni]))
        if use_cuda:
            decoder_input = decoder_input.cuda()        
        if ni == 1:
            break
    print("Q: {}, A: {}".format(q, a))
    print(lang.vectorToSentence(ans))            
            
if __name__ == '__main__':
    main()
