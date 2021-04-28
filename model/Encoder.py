import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.modules import LayerNorm


class RNNBaseEncoder(nn.Module):
    def __init__(self, cell_type,
                 input_size,
                 output_size,
                 num_layers,
                 bidirectional=False,
                 dropout=0.1,
                 layernorm=False):
        super(RNNBaseEncoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        #  if bidirectional:
        #      # 确保output_size = cell_size * directions
        #      assert output_size % 2 == 0
        #      cell_size = output_size // 2
        #  else:
        #      cell_size = output_size

        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.output_size = output_size
        self.num_layers = num_layers
        self.ln = layernorm
        self.layernorm = LayerNorm(self.output_size)
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=output_size,
                                               num_layers=num_layers,
                                               bidirectional=bidirectional,
                                               #  dropout=dropout,
                                               dropout=(0 if num_layers == 1 else dropout),
                                               #  batch_first=True,
                                               )

    def forward(self, x,  # [seq, batch,dim]
                length):  # [batch, ]
        x = pack_padded_sequence(x, length, batch_first=False, enforce_sorted=False)

        # output: [seq, batch,  directions * dim] 每个时间步的隐状态
        # final_state = [layers * directions, batch, dim] 每一层的最后一个状态，不管batch_first是true或false，batch都在中间
        output, final_state = self.rnn_cell(x)
        output = pad_packed_sequence(output, batch_first=False)[0]  # 返回output和length，不需要length了

        if self.bidirectional:
            # 合并双向隐状态
            output = output[:, :, :self.output_size] + output[:, :, self.output_size:]
            if self.ln:
                output = self.layernorm(output)
            if self.cell_type == 'GRU':
                final_state_forward = final_state[0::2, :, :]  # [layers, batch, dim] 偶数的地方是forward
                final_state_backward = final_state[1::2, :, :]  # [layers, batch, dim] 奇数的地方是backward
                #  final_state = torch.cat([final_state_forward, final_state_backward], 2)  # 旧方法，双向时hidden_size是output_size的一半，[layers, batch, 2 * dim = output_size]
                #  合并双向final_state
                final_state = final_state_forward + final_state_backward  # [layers, batch, output_size]
                if self.ln:
                    final_state = self.layernorm(final_state)
            else:
                final_state_h, final_state_c = final_state
                #  final_state_h = torch.cat([final_state_h[0::2, :, :], final_state_h[1::2, :, :]], 2)  # 旧方法，双向时hidden_size是output_size的一半，[layers, batch, 2 * dim = output_size]
                #  final_state_c = torch.cat([final_state_c[0::2, :, :], final_state_c[1::2, :, :]], 2)  # [layers, batch, 2 * dim = output_size]
                #  合并双向的final_state
                final_state_h = final_state_h[0::2, :, :] + final_state_h[1::2, :, :]  # [layers, batch, output_size]
                final_state_c = final_state_c[0::2, :, :] + final_state_c[1::2, :, :]  # [layers, batch, output_size]
                if self.ln:
                    final_state_h = self.layernorm(final_state_h)
                    final_state_c = self.layernorm(final_state_c)
                final_state = (final_state_h, final_state_c)

        # output = [seq, batch, output_size]
        # final_state = [layers, batch, output_size]
        return output, final_state

    def get_output_dim(self):
        return self.output_size
