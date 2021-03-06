import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNBaseEncoder(nn.Module):
    def __init__(self, cell_type,
                 input_size,
                 output_size,
                 num_layers,
                 bidirectional=False,
                 dropout=0.1):
        super(RNNBaseEncoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        if bidirectional:
            # 确保output_size = cell_size * directions
            assert output_size % 2 == 0
            cell_size = output_size // 2
        else:
            cell_size = output_size

        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.output_size = output_size
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=cell_size,
                                               num_layers=num_layers,
                                               bidirectional=bidirectional,
                                               dropout=dropout,
                                               batch_first=True)

    def forward(self, x,  # [batch, seq, dim]
                length):  # [batch, ]
        x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)

        # output: [batch, seq,  directions * dim] 每个时间步的隐状态
        # final_state = [layers * directions, batch, dim] 每一层的最后一个状态，不管batch_first是true或false，batch都在中间
        output, final_state = self.rnn_cell(x)
        output = pad_packed_sequence(output, batch_first=True)[0]  # 返回output和length，不需要length了

        if self.bidirectional:
            if self.cell_type == 'GRU':
                final_state_forward = final_state[0::2, :, :]  # [layers, batch, dim] 偶数的地方是forward
                final_state_backward = final_state[1::2, :, :]  # [layers, batch, dim] 奇数的地方是backward
                final_state = torch.cat([final_state_forward, final_state_backward], 2)  # [layers, batch, 2 * dim = output_size]
            else:
                final_state_h, final_state_c = final_state
                final_state_h = torch.cat([final_state_h[0::2, :, :], final_state_h[1::2, :, :]], 2)  # [layers, batch, 2 * dim = output_size]
                final_state_c = torch.cat([final_state_c[0::2, :, :], final_state_c[1::2, :, :]], 2)  # [layers, batch, 2 * dim = output_size]
                final_state = (final_state_h, final_state_c)

        # output = [batch, seq, output_size]
        # final_state = [layers, batch, directions * dim] = [layers, batch, output_size]
        return output, final_state

    def get_output_dim(self):
        return self.output_size
