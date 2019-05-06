import torch
from torch import nn
import torch.nn.functional as F

class MultiLayerLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bias=True):
        super(MultiLayerLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(self.input_size, self.hidden_size, self.bias)])
        self.lstm_cells.extend(nn.ModuleList(
            nn.LSTMCell(self.hidden_size, self.hidden_size, self.bias)
            for _ in range(self.num_layers - 1)
        ))

    def forward(self, input, states):
        # input: Tensor (batch_size, input_size)
        # states: (hidden, cell)
        # hidden: Tensor (num_layers, batch_size, hidden_size)
        # cell: Tensor (num_layers, batch_size, hidden_size)
        hidden, cell = states
        output_hidden = []
        output_cell = []
        for i, lstm_cell in enumerate(self.lstm_cells):
            h, c = lstm_cell(input, (hidden[i], cell[i]))
            output_hidden.append(h)
            output_cell.append(c)
            input = F.dropout(h, p=self.dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        output_cell = torch.stack(output_cell, dim=0)
        return output_hidden, output_cell


class MultiLayerGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bias=True):
        super(MultiLayerGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.gru_cells = nn.ModuleList([nn.GRUCell(self.input_size, self.hidden_size, self.bias)])
        self.gru_cells.extend(nn.ModuleList(
            nn.GRUCell(self.hidden_size, self.hidden_size, self.bias)
            for _ in range(self.num_layers - 1)
        ))

    def forward(self, input, states):
        # input: Tensor (batch_size, input_size)
        # states: hidden
        # hidden: Tensor (num_layers, batch_size, hidden_size)
        hidden = states
        output_hidden = []
        for i, gru_cell in enumerate(self.gru_cells):
            h = gru_cell(input, hidden[i])
            output_hidden.append(h)
            input = F.dropout(h, p=self.dropout, training=self.training)
        output_hidden = torch.stack(output_hidden, dim=0)
        return output_hidden