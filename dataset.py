import torch
from torch.utils.data import Dataset
import numpy as np
import operator
from model.utils import PAD, SOS, EOS, UNK

class Vocab(object):

    def __init__(self):
        self._count_dict = dict()
        self._predefined_list = [PAD, SOS, EOS, UNK]

    def add(self, word):
        if word in self._count_dict:
            self._count_dict[word] += 1
        else:
            self._count_dict[word] = 1

    def add_list(self, words):
        for word in words:
            self.add(word)

    def get_vocab(self, max_size=None, min_freq=0):
        sorted_words = sorted(self._count_dict.items(), key=operator.itemgetter(1), reverse=True)
        word2index = {}
        for word in self._predefined_list:
            word2index[word] = len(word2index)
        for word, freq in sorted_words:
            if (max_size is not None and len(word2index) >= max_size) or freq < min_freq:
                word2index[word] = word2index[UNK]
            else:
                word2index[word] = len(word2index)
        index2word = {}
        index2word[word2index[UNK]] = UNK
        for word, index in word2index.items():
            if index == word2index[UNK]:
                continue
            else:
                index2word[index] = word
        return word2index, index2word

class Seq2SeqDataset(Dataset):

    def __init__(self, path):
        super(Seq2SeqDataset, self).__init__()
        data = np.load(path)
        self.src = torch.tensor(data['src']).long()
        # self._src_lens = torch.tensor(data['src_lens']).long()
        self.trg = torch.tensor(data['trg']).long()
        # self._trg_lens = torch.tensor(data['trg_lens']).long()
        self.len = self.src.size(0)

    def __getitem__(self, index):
        return self.src[index], self.trg[index]

    def __len__(self):
        return self.len