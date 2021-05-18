import torch.nn as nn
from utils import init_weights
import torch
import random


class RNNBaseSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, layernorm=None):
        super(RNNBaseSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.cell_type == decoder.cell_type
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.tgt_vocab_size = generator.get_output_dim()
        self.layernorm = layernorm
        self.concat1 = nn.Linear(encoder.get_output_dim(), decoder.get_output_dim())
        if encoder.cell_type == 'LSTM':
            self.concat2 = nn.Linear(encoder.get_output_dim(), decoder.get_output_dim())
        self.apply(init_weights)

    def forward_parallel(self, src, src_len, tgt):
        """默认完全使用teacher forcing训练，易导致过拟合训练集"""
        encoder_output, final_state = self.encode(src, src_len)
        if self.encoder.cell_type == 'GRU':
            final_state = self.concat1(final_state[-self.decoder.num_layers:])
        else:
            final_state = list(final_state)
            final_state[0] = self.concat1(final_state[0][-self.decoder.num_layers:])
            final_state[1] = self.concat2(final_state[1][-self.decoder.num_layers:])
            final_state = tuple(final_state)
        output, final_state = self.decode(tgt, final_state, encoder_output)
        logits = self.generator(output)
        # 原本的logits[i]预测的是第i + 1个词，这里为了与forward中保持一致，即第logits[i]预测第i个单词，往右偏移一位。
        logits[1:] = logits.clone()[:-1]
        return logits

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=1):
        if teacher_forcing_ratio == 1 and self.decoder.attn is None:
            return self.forward_parallel(src, src_len, tgt)
        # src = [src len, batch size]
        # tgt = [tgt len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]

        # tensor to store decoder outputs
        logits = torch.zeros(tgt_len, batch_size, self.tgt_vocab_size).to(src.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # output = [seq, batch size, output_size]
        # final_state = [layers, batch size, output_size]
        encoder_output, final_state = self.encode(src, src_len)
        if self.encoder.cell_type == 'GRU':
            # 把encoder的最后decoder.num_layers层最后一步的隐状态作为decoder的起始状态
            final_state = self.concat1(final_state[-self.decoder.num_layers:])
        else:
            final_state = list(final_state)
            # 把encoder的最后decoder.num_layers层最后一步的隐状态作为decoder的起始状态
            final_state[0] = self.concat1(final_state[0][-self.decoder.num_layers:])
            final_state[1] = self.concat2(final_state[1][-self.decoder.num_layers:])
            #  final_state[0] = final_state[0][-self.decoder.num_layers:]
            #  final_state[1] = final_state[1][-self.decoder.num_layers:]
            final_state = tuple(final_state)

        # first input to the decoder is the <sos> tokens
        # inputs = [1, batch size]
        inputs = tgt[0, :].unsqueeze(0)

        for t in range(1, tgt_len):
            # output: [1, batch size, dim]
            # final_state: [layers, batch size, dim]
            output, final_state = self.decode(inputs, final_state, encoder_output)
            if self.layernorm:
                output = self.layernorm(output)
            # predictions = [batch size, tgt_vocab_size]
            predictions = self.generator(output).squeeze(0)
            logits[t] = predictions
            # top1 = [1, batch size]
            top1 = predictions.argmax(1).unsqueeze(0)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inputs = tgt[t].unsqueeze(0) if teacher_force else top1

        return logits

    def encode(self, src, src_len):
        return self.encoder(self.src_embed(src), src_len)

    def decode(self, tgt, state, encoder_output):
        return self.decoder(self.tgt_embed(tgt), state, encoder_output)



class RNNBasePMISeq2Seq(nn.Module):
    def __init__(self, pmi_hid_dim, encoder, decoder, src_embed, tgt_embed, generator, layernorm=None):
        super(RNNBasePMISeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.cell_type == decoder.cell_type
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.vf_mlp = nn.Linear(generator.get_output_dim(), pmi_hid_dim)
        self.generator = generator
        self.tgt_vocab_size = generator.get_output_dim()
        self.layernorm = layernorm
        self.apply(init_weights)

    def forward(self, src, src_len, tgt, pmi, teacher_forcing_ratio=1):
        if teacher_forcing_ratio == 1 and self.decoder.attn is None:
            return self.forward_parallel(src, src_len, tgt, pmi=pmi)
        # src = [src len, batch size]
        # tgt = [tgt len, batch size]
        # pmi = [batch size, tgt vocab size]
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]

        # tensor to store decoder outputs
        logits = torch.zeros(tgt_len, batch_size, self.tgt_vocab_size).to(src.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # output = [seq, batch size, output_size]
        # final_state = [layers, batch size, output_size]
        encoder_output, final_state = self.encode(src, src_len)
        
        # pmi = [1, batch size, tgt vocab size]
        pmi = pmi.unsqueeze(0)

        if self.layernorm is not None:
            if self.layernorm.conditional:
                ln_input = (encoder_output, pmi)
            else:
                ln_input = encoder_output
            encoder_output = self.layernorm(ln_input)

        # pmi = [1, batch size, pmi_hid_dim]
        pmi = self.vf_mlp(pmi)
        final_state = self.concat(final_state, pmi)

        # first input to the decoder is the <sos> tokens
        # inputs = [1, batch size]
        inputs = tgt[0, :].unsqueeze(0)

        for t in range(1, tgt_len):
            # output: [1, batch size, dim]
            # final_state: [layers, batch size, dim]
            output, final_state = self.decode(inputs, final_state, encoder_output)
            # predictions = [batch size, tgt_vocab_size]
            predictions = self.generator(output).squeeze(0)
            logits[t] = predictions
            # top1 = [1, batch size]
            top1 = predictions.argmax(1).unsqueeze(0)

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            inputs = tgt[t].unsqueeze(0) if teacher_force else top1

        return logits

    def encode(self, src, src_len):
        return self.encoder(self.src_embed(src), src_len)

    def decode(self, tgt, state, encoder_output):
        return self.decoder(self.tgt_embed(tgt), state, encoder_output)

    def concat(self, final_state, other):
        if self.decoder.num_layers > 1:
            other = other.repeat(self.decoder.num_layers, 1, 1)
        if self.encoder.cell_type == 'GRU':
            # 把encoder的最后decoder.num_layers层最后一步的隐状态作为decoder的起始状态
            final_state = final_state[-self.decoder.num_layers:]
            final_state = torch.cat((final_state, other), dim=-1)
        else:
            final_state = list(final_state)
            # 把encoder的最后decoder.num_layers层最后一步的隐状态作为decoder的起始状态
            final_state[0] = final_state[0][-self.decoder.num_layers:]
            final_state[1] = final_state[1][-self.decoder.num_layers:]
            final_state[0] = torch.cat((final_state[0], other), dim=-1)
            final_state[1] = torch.cat((final_state[1], other), dim=-1)
            final_state = tuple(final_state)

        return final_state


class RNNBasePMISeq2Seq_v4(nn.Module):
    """v4与v3的区别在于把decoder output通过一个linear层计算lamda，lamda作为gate，决定要给prediction加权多少pmi"""
    def __init__(self, pmi_hid_dim, encoder, decoder, src_embed, tgt_embed, generator, layernorm=None):
        super(RNNBasePMISeq2Seq_v4, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.cell_type == decoder.cell_type
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.vf_mlp = nn.Sequential(nn.Linear(generator.get_output_dim(), pmi_hid_dim),
                                    nn.ReLU())
        self.generator = generator
        self.tgt_vocab_size = generator.get_output_dim()
        self.layernorm = layernorm
        self.lamda = nn.Sequential(nn.Linear(self.decoder.get_output_dim(), 1),
                                nn.Sigmoid())
        self.concat1 = nn.Linear(self.encoder.get_output_dim() * 2, self.decoder.get_output_dim())
        self.concat2 = nn.Linear(self.encoder.get_output_dim() * 2, self.decoder.get_output_dim())

        self.apply(init_weights)

    def forward(self, src, src_len, tgt, pmi, teacher_forcing_ratio=1):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]

        logits = torch.zeros(tgt_len, batch_size, self.tgt_vocab_size).to(src.device)

        encoder_output, final_state = self.encode(src, src_len)
        
        # pmi = [1, batch size, tgt vocab size]
        pmi = pmi.unsqueeze(0)
        pmi_origin = pmi.clone()

        # pmi = [1, batch size, pmi_hid_dim]
        pmi = self.vf_mlp(pmi)
        final_state = self.concat(final_state, pmi)

        inputs = tgt[0, :].unsqueeze(0)

        for t in range(1, tgt_len):
            output, final_state = self.decode(inputs, final_state, encoder_output)
            # lamda = [batch size, 1]
            lamda = self.lamda(output).squeeze(0)
            
            if self.layernorm is not None:
                if self.layernorm.conditional:
                    ln_input = (output, pmi_origin)
                else:
                    ln_input = output
                output = self.layernorm(ln_input)
            # predictions = [batch size, tgt_vocab_size]
            predictions = self.generator(output).squeeze(0)
            predictions = (1 - lamda) * predictions + lamda * pmi_origin.squeeze(0)
            logits[t] = predictions
            # top1 = [1, batch size]
            top1 = predictions.argmax(1).unsqueeze(0)

            teacher_force = random.random() < teacher_forcing_ratio

            inputs = tgt[t].unsqueeze(0) if teacher_force else top1

        return logits

    def encode(self, src, src_len):
        return self.encoder(self.src_embed(src), src_len)

    def decode(self, tgt, state, encoder_output):
        return self.decoder(self.tgt_embed(tgt), state, encoder_output)

    def concat(self, final_state, other):
        if self.decoder.num_layers > 1:
            other = other.repeat(self.decoder.num_layers, 1, 1)
        if self.encoder.cell_type == 'GRU':
            final_state = final_state[-self.decoder.num_layers:]
            final_state = torch.cat((final_state, other), dim=-1)
        else:
            final_state = list(final_state)
            final_state[0] = final_state[0][-self.decoder.num_layers:]
            final_state[1] = final_state[1][-self.decoder.num_layers:]
            final_state[0] = torch.cat((final_state[0], other), dim=-1)
            final_state[1] = torch.cat((final_state[1], other), dim=-1)
            final_state[0] = self.concat1(final_state[0])
            final_state[1] = self.concat2(final_state[1])
            final_state = tuple(final_state)

        return final_state


class RNNBasePMISeq2Seq_v5(nn.Module):
    """v5中使用encoder最后一层输出的final hidden state去预测PMI，最小化KL divergence loss, 因此final state不再与pmi concat
    然后作为decoder的init state"""
    def __init__(self, pmi_hid_dim, encoder, decoder, src_embed, tgt_embed, generator, layernorm=None):
        super(RNNBasePMISeq2Seq_v5, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.cell_type == decoder.cell_type
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        #  self.vf_mlp = nn.Linear(generator.get_output_dim(), pmi_hid_dim)
        self.generator = generator
        self.tgt_vocab_size = generator.get_output_dim()
        self.layernorm = layernorm
        self.lamda = nn.Sequential(nn.Linear(self.decoder.get_output_dim(), 1),
                                nn.Sigmoid())
        #  self.concat1 = nn.Linear(self.encoder.get_output_dim() * 2, self.decoder.get_output_dim())
        #  self.concat2 = nn.Linear(self.encoder.get_output_dim() * 2, self.decoder.get_output_dim())
        self.pred_kl_logits = nn.Linear(self.encoder.get_output_dim(), self.tgt_vocab_size)

        # self.apply(init_weights)

    def forward(self, src, src_len, tgt, pmi, teacher_forcing_ratio=1):
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]

        logits = torch.zeros(tgt_len, batch_size, self.tgt_vocab_size).to(src.device)

        encoder_output, final_state = self.encode(src, src_len)
        # kl_logits = [batch size, tgt vocab size]
        kl_logits = self.pred_kl_logits(final_state[0][-1])
        
        # pmi = [1, batch size, tgt vocab size]
        pmi = pmi.unsqueeze(0)
        pmi_origin = pmi.clone()

        # pmi = [1, batch size, pmi_hid_dim]
        #  pmi = self.vf_mlp(pmi)
        #  final_state = self.concat(final_state, pmi)

        inputs = tgt[0, :].unsqueeze(0)

        for t in range(1, tgt_len):
            output, final_state = self.decode(inputs, final_state, encoder_output)
            # lamda = [batch size, 1]
            lamda = self.lamda(output).squeeze(0)
            
            if self.layernorm is not None:
                if self.layernorm.conditional:
                    ln_input = (output, pmi_origin)
                else:
                    ln_input = output
                output = self.layernorm(ln_input)
            # predictions = [batch size, tgt_vocab_size]
            predictions = self.generator(output).squeeze(0)
            predictions = (1 - lamda) * predictions + lamda * pmi_origin.squeeze(0)
            logits[t] = predictions
            # top1 = [1, batch size]
            top1 = predictions.argmax(1).unsqueeze(0)

            teacher_force = random.random() < teacher_forcing_ratio

            inputs = tgt[t].unsqueeze(0) if teacher_force else top1

        return logits, kl_logits

    def encode(self, src, src_len):
        return self.encoder(self.src_embed(src), src_len)

    def decode(self, tgt, state, encoder_output):
        return self.decoder(self.tgt_embed(tgt), state, encoder_output)

    def concat(self, final_state, other):
        if self.decoder.num_layers > 1:
            other = other.repeat(self.decoder.num_layers, 1, 1)
        if self.encoder.cell_type == 'GRU':
            final_state = final_state[-self.decoder.num_layers:]
            final_state = torch.cat((final_state, other), dim=-1)
        else:
            final_state = list(final_state)
            final_state[0] = final_state[0][-self.decoder.num_layers:]
            final_state[1] = final_state[1][-self.decoder.num_layers:]
            final_state[0] = torch.cat((final_state[0], other), dim=-1)
            final_state[1] = torch.cat((final_state[1], other), dim=-1)
            final_state[0] = self.concat1(final_state[0])
            final_state[1] = self.concat2(final_state[1])
            final_state = tuple(final_state)

        return final_state


