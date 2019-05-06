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
        src = sentence_clip(src)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src_memory, init_states = self.encode(src, src_lens)
        init_output = self.decoder.get_init_output(src_memory, src_lens, init_states)
        return self.decoder(src_memory, src_mask, init_states, init_output, trg)

    def encode(self, src, src_lens):
        """
        :param src: LongTensor (batch_size, time_step)
        :param src_lens: LongTensor (batch_size,)
        :return:
        """
        src_embedding = self.embedding(src)   # Tensor(batch_size, time_step, embed_size)
        src_memory, final_states = self.encoder(src_embedding, src_lens)
        return src_memory, final_states

    def decode(self, src, max_len):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param max_len: int
        :return:
        """
        src = sentence_clip(src)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src_memory, init_states = self.encode(src, src_lens)
        init_output = self.decoder.get_init_output(src_memory, src_lens, init_states)
        batch_size = src_memory.size(0)
        token = torch.tensor([SOS_INDEX] * batch_size).cuda()
        states = init_states
        output = init_output
        outputs = []
        for _ in range(max_len):
            logit, states, output = self.decoder.step(src_memory, src_mask, token, states, output)
            token = torch.argmax(logit, dim=1, keepdim=False)
            outputs.append(token)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def beam_decode(self, src, max_len, beam_size):
        """
        :param src: LongTensor (batch_size, src_time_step)
        :param max_len: int
        :param beam_size: int
        :return:
        """
        src = sentence_clip(src)
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        src_memory, init_states = self.encode(src, src_lens)
        init_output = self.decoder.get_init_output(src_memory, src_lens, init_states)
        batch_size, time_step, hidden_size = src_memory.size()
        src_memory = src_memory.repeat(beam_size, 1, 1, 1).view(beam_size * batch_size, time_step, hidden_size).contiguous()
        src_mask = src_mask.repeat(beam_size, 1, 1).view(beam_size * batch_size, time_step).contiguous()
        beamer = Beamer(
            states=init_states,
            output=init_output,
            beam_size=beam_size,
            remove_repeat_triple_grams=True
        )
        for _ in range(max_len):
            token, states, output = beamer.pack_batch()
            logit, states, output = self._decoder.step(
                src_memory, src_mask, token, states, output
            )
            log_prob = F.log_softmax(logit, dim=-1)
            log_prob, token = log_prob.topk(k=beam_size, dim=-1)
            beamer.update_beam(token, log_prob, states, output)
        outputs = beamer.get_best_sequences(max_len)
        return outputs
