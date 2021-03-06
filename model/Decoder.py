import copy
import torch.nn as nn
import torch.nn.functional as F
from utils import clones


class RNNBaseDecoder(nn.Module):
    def __init__(self, cell_type,
                 input_size,
                 output_size,
                 num_layers,
                 dropout=0.1
                 ):
        super(RNNBaseDecoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        self.cell_type = cell_type
        self.output_size = output_size
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=output_size,
                                               num_layers=num_layers,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x,  # x = [batch, seq, dim] 或单步输入 [batch, 1, dim]
                state):  # state = [layers, batch, dim]
        # decoder不存在双向，所以directions永远是1，因此省略
        # output: [batch, seq, dim] 每个时间步的输出
        # final_state: [batch, layers, dim] 每一层的最终状态
        output, final_state = self.rnn_cell(x, state)
        return output, final_state

    def get_output_dim(self):
        return self.output_size


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab, bias=False)
        self.vocab = vocab

    def forward(self, x):
        return self.proj(x)

    def get_output_dim(self):
        return self.vocab
