from chatTool import *
from Net import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import optim
import pickle
import argparse

use_cuda = torch.cuda.is_available()

LangBag = "dict.pkl"

with open(LangBag, 'rb') as f:
    lang = pickle.load(f)
    
parser = argparse.ArgumentParser()
parser.add_argument('-em', help="Encoder model Name", required=True)
parser.add_argument('-dm', help="Decoder model Name", required=True)
parser.add_argument('-q', help="Question", required=True)

args = parser.parse_args()

EncoderFile = args.em#"Encoder.pth"
DecoderFile = args.dm#"Decoder.pth"

INPUTSIZE = len(lang.index2word)

if os.path.isfile(EncoderFile):
    encoder = torch.load(EncoderFile)
    print("Load encoder model: {}.".format(EncoderFile))
else:
    encoder = EncoderRNN(INPUTSIZE, 64, 256)
    print("Create encoder model.")

if os.path.isfile(DecoderFile):
    decoder = torch.load(DecoderFile)
    print("Load decoder model: {}.".format(DecoderFile))
else:
    decoder = DecoderRNN(INPUTSIZE, 64, 256, INPUTSIZE)
    print("Create decoder model.")

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
def main():
    print("Q: {}".format(args.q))
    print("A: {}".format(predit(args.q)))
def predit(q):
    input = torch.Tensor(lang.sentenceToVector(q)).long()
    input = Variable(input)
    if use_cuda:
        input = input.cuda()
    encoder_hidden = encoder.initHidden()
    for ei in input:
        o, encoder_hidden = encoder(ei, encoder_hidden)
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
    return lang.vectorToSentence(ans)
    
if __name__ == '__main__':
    main()  