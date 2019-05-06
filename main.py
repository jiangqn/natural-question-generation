# -*- coding: utf-8 -*-
# The foregoing line is for telling python to interpret the following string with utf-8.（如果文件中有中文必须有上一条）
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from model.seq2seq import Seq2Seq
from dataset import Seq2SeqDataset
from model.utils import EOS_INDEX, PAD_INDEX, sentence_clip, tokenize
from model.criterion import WordCrossEntropy, SentenceCrossEntropy
import pickle
from nltk.translate.bleu_score import corpus_bleu

# The following five lines are for reproducibility. https://pytorch.org/docs/master/notes/randomness.html
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class Trainer(object):

    """To build training instance.

    Use the Trainer.run() method to train.

    """

    def __init__(self, config):
        self._config = config

    def make_model(self):
        model = Seq2Seq(
            vocab_size=self._config.vocab_size,
            embed_size=self._config.embed_size,
            hidden_size=self._config.hidden_size,
            rnn_type=self._config.rnn_type,
            num_layers=self._config.num_layers,
            bidirectional=self._config.bidirectional,
            attention_type=self._config.attention_type,
            dropout=self._config.dropout
        )
        model.load_pretrained_embeddings(self._config.embedding_file_name)
        return model

    def make_data(self):
        train_dataset = Seq2SeqDataset(self._config.train_path)
        dev_dataset = Seq2SeqDataset(self._config.dev_path)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=2
        )
        dev_loader = DataLoader(
            dataset=dev_dataset,
            batch_size=self._config.batch_size,
            shuffle=False,
            num_workers=2
        )
        return train_loader, dev_loader

    def make_vocab(self):
        with open(self._config.vocab_path, 'rb') as handle:
            self._index2word = pickle.load(handle)

    def run(self):
        self.make_vocab()
        model = self.make_model()
        model = model.cuda()
        print(model)
        criterion = SentenceCrossEntropy()
        optimizer = optim.Adam(model.parameters(), lr=self._config.learning_rate)

        train_loader, dev_loader = self.make_data()

        for epoch in range(1, self._config.num_epoches + 1):
            sum_loss = 0
            sum_examples = 0
            model.train()
            for i, data in enumerate(train_loader):
                src, trg = data
                src, trg = src.cuda(), trg.cuda()
                trg = sentence_clip(trg)
                optimizer.zero_grad()
                logits = model(src, trg[:, 0: -1].contiguous())
                loss = criterion(logits, trg[:, 1:].contiguous())
                sum_loss += loss.item() * src.size(0)
                sum_examples += src.size(0)
                if i > 0 and i % 100 == 0:
                    print('[epoch %2d] [step %4d] [loss %.4f]' % (epoch, i, sum_loss / sum_examples))
                    sum_loss = 0
                    sum_examples = 0
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._config.clip)
                optimizer.step()
            self.eval(model, dev_loader, epoch)
            self.save_model(model, epoch)

    def ndarray2texts(self, ndarray):
        texts = []
        for vector in ndarray:
            text = ''
            for index in vector:
                if index == EOS_INDEX or index == PAD_INDEX:
                    break
                text += self._index2word[index] + ' '
            texts.append(text.strip())
        return texts

    def eval(self, model, data_loader, epoch=None):
        model.eval()
        pred = []
        for data in data_loader:
            src, trg = data
            trg_mask = trg != PAD_INDEX
            trg_lens = trg_mask.long().sum(dim=1, keepdim=False)
            src = src.cuda()
            with torch.no_grad():
                output = model.decode(src, trg_lens.max().item() + 1)
                texts = self.ndarray2texts(output.cpu().numpy())
                print(texts[0])
                pred.extend(texts)
        path = './data/output/pred' + (('-epoch-' + str(epoch)) if epoch is not None else '') + '.txt'
        self.write_file(pred, path)
        bleu = self.calculate_bleu(path, self._config.dev_reference_path)
        print('bleu: %.4f' % bleu)

    def write_file(self, texts, path):
        file = open(path, 'w', encoding=u'utf-8')
        for text in texts:
            file.write(text + '\n')

    def calculate_bleu(self, hypotheses_path, references_path):
        hypotheses_file = open(hypotheses_path, 'r', encoding='utf-8')
        references_file = open(references_path, 'r', encoding='utf-8')
        hyp_trans = lambda x: tokenize(x.strip())
        ref_trans = lambda x: [tokenize(x.strip())]
        hypotheses = list(map(hyp_trans, hypotheses_file.readlines()))
        references = list(map(ref_trans, references_file.readlines()))
        bleu = corpus_bleu(references, hypotheses, weights=[1, 0, 0, 0])
        return bleu

    def save_model(self, model, epoch=None):
        path = './data/checkpoints/model' + (('-epoch-' + str(epoch)) if epoch is not None else '') + '.pkl'
        torch.save(model, path)


parser = argparse.ArgumentParser()
parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--attention_type', type=str, default='Bilinear', choices=['Dot', 'ScaledDot', 'Concat', 'Bilinear', 'MLP'])
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--vocab_size', type=int, default=37411)
parser.add_argument('--hidden_size', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--bidirectional', type=bool, default=True)
parser.add_argument('--num_epoches', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--l2_reg', type=float, default=0)
parser.add_argument('--clip', type=float, default=5.0)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--embedding_file_name', type=str, default='data/processed/glove.npy')
parser.add_argument('--vocab_path', type=str, default='./data/processed/index2word.pkl')
parser.add_argument('--train_path', type=str, default='./data/processed/train.npz')
parser.add_argument('--dev_path', type=str, default='./data/processed/dev.npz')
parser.add_argument('--dev_reference_path', type=str, default='./data/raw/trg_dev.txt')
parser.add_argument('--gpu_id', type=str, default='5')

config = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

trainer = Trainer(config)
trainer.run()
