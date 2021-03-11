import torch.nn as nn
from utils import init_weights
import torch
import random


class RNNBaseSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(RNNBaseSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.cell_type == decoder.cell_type
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        # self.apply(init_weights)

    def forward_parallel(self, src, src_len, tgt):
        """默认完全使用teacher forcing训练，易导致过拟合训练集"""
        output, final_state = self.decode(tgt, self.encode(src, src_len)[1])
        return self.generator(output)

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # tgt = [tgt len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = tgt.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.generator.get_output_dim()

        # tensor to store decoder outputs
        logits = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(src.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        # output = [seq, batch size, output_size]
        # final_state = [layers, batch size, output_size]
        output, final_state = self.encode(src, src_len)

        # first input to the decoder is the <sos> tokens
        # inputs = [1, batch size]
        inputs = tgt[0, :].unsqueeze(0)

        for t in range(1, tgt_len):
            # output: [1, batch size, dim]
            # final_state: [layers, batch size, dim]
            output, final_state = self.decode(inputs, final_state)
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

    def decode(self, tgt, state):
        return self.decoder(self.tgt_embed(tgt), state)
