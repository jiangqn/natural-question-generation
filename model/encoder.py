import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):

    def __init__(self, embed_size, hidden_size, rnn_type='LSTM', num_layers=1, bidirectional=False, dropout=0):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        state_layers = num_layers * (2 if bidirectional else 1)
        output_size = hidden_size * (2 if bidirectional else 1)
        self.output_projection = nn.Linear(output_size, hidden_size)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
            self.init_states = nn.ParameterList([
                nn.Parameter(
                    torch.Tensor(state_layers, hidden_size)
                ),
                nn.Parameter(
                    torch.Tensor(state_layers, hidden_size)
                )
            ])
            self.hidden_projection = nn.Linear(output_size, hidden_size)
            self.cell_projection = nn.Linear(output_size, hidden_size)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True
            )
            self.init_states = nn.Parameter(
                torch.Tensor(state_layers, hidden_size)
            )
            self.hidden_projection = nn.Linear(output_size, hidden_size)
        else:
            raise ValueError('%s is not supported.' % str(rnn_type))
        self._reset_parameters()

    def _reset_parameters(self):
        if self.rnn_type == 'LSTM':
            init.xavier_uniform_(self.init_states[0])
            init.xavier_uniform_(self.init_states[1])
        else:
            init.xavier_uniform_(self.init_states)

    def forward(self, src_embedding, src_lens):
        """
        :param src_embedding: FloatTensor (batch_size, src_time_step, embed_size)
        :param src_lens: LongTensor (batch_size,)
        :return:
        """
        batch_size = src_embedding.size(0)
        init_states = self._get_init_states(batch_size)
        src_lens, sort_index = src_lens.sort(descending=True)
        src_embedding = src_embedding.index_select(dim=0, index=sort_index)
        packed_src = pack_padded_sequence(src_embedding, src_lens, batch_first=True)
        packed_output, final_states = self.rnn(packed_src, init_states)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        sort_index = sort_index.argsort(descending=False)
        output = output.index_select(index=sort_index, dim=0)
        if self.rnn_type == 'LSTM':  # LSTM
            final_states = (
                final_states[0].index_select(index=sort_index, dim=1),
                final_states[1].index_select(index=sort_index, dim=1)
            )
        else:  # GRU
            final_states = final_states.index_select(index=sort_index, dim=1)
        # bridge
        output = self.output_projection(output)
        if self.rnn_type == 'LSTM':  # LSTM
            if self.bidirectional:
                final_states = (
                    torch.cat(
                        final_states[0].chunk(chunks=2, dim=0),
                        dim=2
                    ),
                    torch.cat(
                        final_states[1].chunk(chunks=2, dim=0),
                        dim=2
                    )
                )
            final_states = (
                torch.stack([
                    self.hidden_projection(hidden) for hidden in final_states[0]
                ], dim=0),
                torch.stack([
                    self.cell_projection(cell) for cell in final_states[1]
                ], dim=0)
            )
        else:  # GRU
            if self.bidirectional:
                final_states = torch.cat(
                    final_states.chunk(chunks=2, dim=0),
                    dim=2
                )
            final_states = torch.stack([
                self.hidden_projection(hidden) for hidden in final_states
            ], dim=0)
        return output, final_states

    def _get_init_states(self, batch_size):
        """
        :param batch_size: int
        :return:
        """
        if self.rnn_type == 'LSTM':    # LSTM
            state_layers, hidden_size = self.init_states[0].size()
            size = (state_layers, batch_size, hidden_size)
            init_states = (
                self.init_states[0].unsqueeze(1).expand(*size).contiguous(),
                self.init_states[1].unsqueeze(1).expand(*size).contiguous()
            )
        else:   # GRU
            state_layers, hidden_size = self.init_states.size()
            size = (state_layers, batch_size, hidden_size)
            init_states = self.init_states.cuda().unsqueeze(1).expand(*size).contiguous()
        return init_states