import os, glob
import pickle
import torch.utils.data as data
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', help="Conv file", required=True)
parser.add_argument('-o', help="Output file name", required=True)

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS": 0, "EOS": 1}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    def sentenceToVector(self, s, sos = False, eos = False):
        numS = []
        if sos: numS.append(self.word2index["SOS"])
        for w in s:
            numS.append(self.word2index[w])
        if eos:
            numS.append(self.word2index["EOS"])
        return numS
    def vectorToSentence(self, v):
        s = []
        try:
            v.remove(0)
        except:
            pass
        try:
            v.remove(1)
        except:
            pass
        for i in v:
            s.append(self.index2word[i])
        return "".join(s)
    
class Reader(data.Dataset):
    def __init__(self, file):
        with open(file) as f:
            self.sents = f.read().split('\n')
        self.pairs = []
        for i in range(len(self.sents)-1):
            if self.sents[i] is not "E" and self.sents[i+1] is not "E" and len(self.sents[i]) != 0 and len(self.sents[i+1]) != 0:
                self.pairs.append(i)
            self.sents[i] = self.sents[i].replace("M ","")
        print("Load file:{}, Size: {}\n".format(file, len(self.pairs)))
    def __getitem__(self, index):
        sentId = self.pairs[index]
        return self.sents[sentId], self.sents[sentId+1]
    def __len__(self):
        return len(self.pairs)
if __name__=='__main__':
    args = parser.parse_args()
    files = glob.glob(os.path.join(args.d, "*.tconv"))
    print("Read data from {}, file: {}".format(args.d, len(files)))

    lmap = Lang(args.d)
    
    for fname in files:
        s=open(fname).read()
        for sent in s.split('\n'):
            if sent is not "E":
                sent=sent.replace("M ","")
                for word in list(sent):
                    lmap.addWord(word)
    with open(args.o, 'wb') as f:
        pickle.dump(lmap, f)
    print("Total word {}.".format(len(lmap.word2index)))
    print("Save model as {}.".format(args.o))
