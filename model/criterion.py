import torch
from torch import nn
from model.utils import PAD_INDEX

class SentenceCrossEntropy(nn.Module):

    def __init__(self):
        super(SentenceCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=PAD_INDEX, reduction='none')

    def forward(self, logits, trg):
        """
        :param logits: (batch_size, time_step, vocab_size)
        :param trg: (batch_size, time_step)
        :return loss: FloatTensor ()
        """
        logits = logits.transpose(1, 2)
        losses = self.cross_entropy(logits, trg)
        trg_mask = trg != PAD_INDEX
        trg_lens = trg_mask.long().sum(dim=1, keepdim=False)
        losses = losses.masked_fill(trg_mask==0, 0)
        losses = losses.sum(dim=1, keepdim=False)
        losses = losses / trg_lens.float()
        loss = losses.mean()
        return loss


class WordCrossEntropy(nn.Module):

    def __init__(self):
        super(WordCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)

    def forward(self, logits, trg):
        """
        :param logits: (batch_size, time_step, vocab_size)
        :param trg: (batch_size, time_step)
        :return loss: FloatTensor ()
        """
        logits = logits.transpose(1, 2)
        return self.cross_entropy(logits, trg)