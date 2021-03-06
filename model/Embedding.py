import torch.nn as nn
import math


class Embedding(nn.Module):
    def __init__(self, num_vocab,
                 embedding_size,
                 padding_idx=0,
                 dropout=0.1,
                 ):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.lut = nn.Embedding(num_embeddings=num_vocab,
                                embedding_dim=embedding_size,
                                padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # [batch, seq]
        return self.dropout(self.lut(x)) #* math.sqrt(self.embedding_size)  # [batch, seq, embedding_size]
