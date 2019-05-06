import numpy as np
from model.utils import tokenize, load_word_embeddings
from dataset import Vocab
from logger import Logger
import pickle
from model.utils import SOS, EOS

train_src_path = './data/raw/src_train.txt'
train_trg_path = './data/raw/trg_train.txt'
dev_src_path = './data/raw/src_dev.txt'
dev_trg_path = './data/raw/trg_dev.txt'
test_src_path = './data/raw/src_test.txt'
test_trg_path = './data/raw/trg_test.txt'

data_log_path = './data/log/data_log.txt'

train_save_path = './data/processed/train.npz'
dev_save_path = './data/processed/dev.npz'
test_save_path = './data/processed/test.npz'

glove_path = './data/vocab/glove.840B.300d.txt'
glove_save_path = './data/processed/glove.npy'

word2index_path = './data/processed/word2index.pkl'
index2word_path = './data/processed/index2word.pkl'

log = Logger(data_log_path)

def process(src_path, trg_path):
    src_max_len, trg_max_len = 0, 0
    src = open(src_path, 'r', encoding=u'utf-8').readlines()
    trg = open(trg_path, 'r', encoding=u'utf-8').readlines()
    assert len(src) == len(trg)
    num = len(src)
    for i in range(num):
        src[i] = tokenize(src[i])
        trg[i] = tokenize(trg[i])
        src_max_len = max(src_max_len, len(src[i]))
        trg_max_len = max(trg_max_len, len(trg[i]))
    return src, trg, num, src_max_len, trg_max_len

train_src, train_trg, train_num, train_src_max_len, train_trg_max_len = process(train_src_path, train_trg_path)
dev_src, dev_trg, dev_num, dev_src_max_len, dev_trg_max_len = process(dev_src_path, dev_trg_path)
test_src, test_trg, test_num, test_src_max_len, test_trg_max_len = process(test_src_path, test_trg_path)

log.write('train_num', train_num)
log.write('train_src_max_len', train_src_max_len)
log.write('train_trg_max_len', train_trg_max_len)
log.write('dev_num', dev_num)
log.write('dev_src_max_len', dev_src_max_len)
log.write('dev_trg_max_len', dev_trg_max_len)
log.write('test_num', test_num)
log.write('test_src_max_len', test_src_max_len)
log.write('test_trg_max_len', test_trg_max_len)

vocab = Vocab()

for i in range(train_num):
    vocab.add_list(train_src[i])
    vocab.add_list(train_trg[i])

for i in range(dev_num):
    vocab.add_list(dev_src[i])
    vocab.add_list(dev_trg[i])

for i in range(test_num):
    vocab.add_list(test_src[i])
    vocab.add_list(test_trg[i])

word2index, index2word = vocab.get_vocab(min_freq=4)
total_words = len(word2index)
vocab_size = len(index2word)

with open(word2index_path, 'wb') as handle:
    pickle.dump(word2index, handle)

with open(index2word_path, 'wb') as handle:
    pickle.dump(index2word, handle)

log.write('total_words', total_words)
log.write('vocab_size', vocab_size)

def text2src(texts, max_len):
    num = len(texts)
    src = np.zeros((num, max_len + 1), dtype=np.int32)
    for i, text in enumerate(texts):
        for j, word in enumerate(text):
            src[i, j] = word2index[word]
        src_len = len(text)
        src[i, src_len] = word2index[EOS]
    return src

def text2trg(texts, max_len):
    num = len(texts)
    trg = np.zeros((num, max_len + 2), dtype=np.int32)
    for i, text in enumerate(texts):
        trg[i, 0] = word2index[SOS]
        for j, word in enumerate(text):
            trg[i, j + 1] = word2index[word]
        trg_len = len(text)
        trg[i, trg_len + 1] = word2index[EOS]
    return trg

train_src = text2src(train_src, train_src_max_len)
train_trg = text2trg(train_trg, train_trg_max_len)
dev_src = text2src(dev_src, dev_src_max_len)
dev_trg = text2trg(dev_trg, dev_trg_max_len)
test_src = text2src(test_src, test_src_max_len)
test_trg = text2trg(test_trg, test_trg_max_len)

np.savez(train_save_path, src=train_src, trg=train_trg)
np.savez(dev_save_path, src=dev_src, trg=dev_trg)
np.savez(test_save_path, src=test_src, trg=test_trg)

glove = load_word_embeddings(glove_path, vocab_size, 300, word2index)
np.save(glove_save_path, glove)