import copy
import torch
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
        self.num_layers = num_layers
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                               hidden_size=output_size,
                                               num_layers=num_layers,
                                               dropout=(0 if num_layers == 1 else dropout),
                                               #  batch_first=True,
                                               )
        self.attn = None

    def forward(self, x,  # x = [seq, batch, dim] 或单步输入 [1, batch, dim]
                state,  # state = [layers, batch, dim]
                encoder_output):
        # decoder不存在双向，所以directions永远是1，因此省略
        # output: [seq, batch, dim] 每个时间步的输出
        # final_state: [layers, batch, dim] 每一层的最终状态
        output, final_state = self.rnn_cell(x, state)
        return output, final_state

    def get_output_dim(self):
        return self.output_size


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, enc_hidden_size, dec_hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.enc_hidden_size, dec_hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(dec_hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        # energy shape = [seq_len, batch_size, dec_hidden_size]
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        # energy shape = [seq_len, batch_size, dec_hidden_size]
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch_size, dec_hidden_size]
        # encoder_outputs = [seq_len, batch_size, enc_hidden_size]
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            assert self.enc_hidden_size == self.dec_hidden_size
            attn_energies = self.dot_score(hidden, encoder_outputs)
        # attn_energies shape = [seq_len, batch_size]

        # Transpose seq_len and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        # return shape = [batch_size, 1, seq_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnRNNDecoder(nn.Module):
    def __init__(self, cell_type, input_size, enc_hidden_size, dec_hidden_size, attn_method, num_layers=1, dropout=0.1):
        super(LuongAttnRNNDecoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']

        # Keep for reference
        self.cell_type = cell_type
        self.attn_method = attn_method
        self.attn = Attn(attn_method, enc_hidden_size, dec_hidden_size)
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Define layers
        self.rnn_cell = getattr(nn, cell_type)(input_size=input_size,
                                            hidden_size=dec_hidden_size,
                                            num_layers=num_layers,
                                            dropout=(0 if num_layers == 1 else dropout),
                                            )
        self.concat = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)

    def forward(self, x, state, encoder_outputs):
        """
        Forward through unidirectional RNN
        x = [1, batch_size, emb_dim]
        encoder_outputs = [seq_len, batch_size, enc_hidden_size]
        """
        # output: [1, batch_size, dim]
        # final_state: [layers, batch, dim]
        output, final_state = self.rnn_cell(x, state)
        # Calculate attention weights from the current RNN output
        # attn_weights shape = [batch_size, 1, seq_len]
        attn_weights = self.attn(output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # context shape = [batch_size, 1, enc_hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        output = output.squeeze(0)
        context = context.squeeze(1)
        # concat_input shape = [batch_size, dec_hidden_size + enc_hidden_size]
        concat_input = torch.cat((output, context), 1)
        # concat_output shape = [1, batch_size, dec_hidden_size]
        concat_output = torch.tanh(self.concat(concat_input)).unsqueeze(0)
        #  有时不用tanh激活函数效果更好
        #  concat_output = (self.concat(concat_input)).unsqueeze(0)
        # Return output and final state
        return concat_output, final_state

    def get_output_dim(self):
        return self.dec_hidden_size


class Generator(nn.Module):
    """Define standard linear generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab, bias=False)
        self.vocab = vocab

    def forward(self, x):
        return self.proj(x)

    def get_output_dim(self):
        return self.vocab
