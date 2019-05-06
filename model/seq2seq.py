import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import SOS_INDEX, PAD_INDEX
from model.beam_search import Beamer
from model.encoder import Encoder
from model.decoder import Decoder
from model.utils import sentence_clip
import numpy as np

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type='LSTM', num_layers=1,
                 bidirectional=False, attention_type='Bilinear', dropout=0):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.encoder = Encoder(
            embed_size=embed_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.decoder = Decoder(
            embedding=self.embedding,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            attention_type=attention_type,
            dropout=dropout
        )

    def load_pretrained_embeddings(self, path):
        self.embedding.weight.data.copy_(torch.from_numpy(np.load(path)))  # Load pretrained embeddings
        self.embedding.weight.requires_grad = False  # Don't update word vectors during training

    def forward(self, src, trg):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param trg: LongTensor (batch_size, trg_time_step)
        :return:
        """
        src_memory, src_mask, src_lens, init_states = self.encode(src)
        init_output = self.decoder.get_init_output(src_memory, src_lens, init_states)
        return self.decoder(src_memory, src_mask, init_states, init_output, trg)

    def encode(self, src):
        """
        :param src: LongTensor (batch_size, time_step)
        :param src_lens: LongTensor (batch_size,)
        :return:
        """
        src = sentence_clip(src)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src_embedding = self.embedding(src)   # Tensor(batch_size, time_step, embed_size)
        src_memory, final_states = self.encoder(src_embedding, src_lens)
        return src_memory, src_mask, src_lens, final_states

    def decode(self, src, max_len):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param max_len: int
        :return:
        """
        src_memory, src_mask, src_lens, init_states = self.encode(src)
        init_output = self.decoder.get_init_output(src_memory, src_lens, init_states)
        outputs = self.decoder.decode(src_memory, src_mask, init_states, init_output, max_len)
        return outputs

    def beam_decode(self, src, max_len, beam_size):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param max_len: int
        :param beam_size: int
        :return:
        """
        src_memory, src_mask, src_lens, init_states = self.encode(src)
        init_output = self.decoder.get_init_output(src_memory, src_lens, init_states)
        outputs = self.decoder.beam_decode(src_memory, src_mask, init_states, init_output, max_len, beam_size)
        return outputs