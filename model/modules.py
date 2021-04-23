import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-12, conditional=False, condition_size=None, condition_hidden_size=None, condition_activation=None):
        super(LayerNorm, self).__init__()
        self.beta = nn.Parameter(torch.ones(feature))
        self.gamma = nn.Parameter(torch.zeros(feature))
        self.eps = eps
        self.conditional = conditional
        self.condition_size = condition_size
        self.condition_hidden_size = condition_hidden_size

        if conditional:
            self.cond_dense = nn.Linear(in_features=condition_size, out_features=condition_hidden_size, bias=False)
            self.beta_dense = nn.Linear(in_features=condition_hidden_size, out_features=feature, bias=False)
            self.gamma_dense = nn.Linear(in_features=condition_hidden_size, out_features=feature, bias=False)
            nn.init.zeros_(self.beta_dense.weight.data)
            nn.init.zeros_(self.gamma_dense.weight.data)
            if condition_activation:
                #  condition_activation must be in nn, for example ['Tanh', 'ReLU', 'Sigmoid']
                self.cond_dense = nn.Sequential(self.cond_dense, getattr(nn, condition_activation)())

    def forward(self, x):
        if self.conditional:
            # x = [seq, batch size, feature]
            # cond = [1, batch size, condition_size]
            x, cond = x
            # cond = [1, batch size, condition_hidden_size]
            cond = self.cond_dense(cond)
            # beta = [1, batch size, feature]
            beta = self.beta_dense(cond) + self.beta
            # gamma = [1, batch size, feature]
            gamma = self.gamma_dense(cond) + self.gamma
        else:
            beta = self.beta
            gamma = self.gamma
        # mean = std = [seq, batch size, 1]
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return beta * (x - mean) / (std + self.eps) + gamma

    def get_output_dim(self):
        return self.feature

