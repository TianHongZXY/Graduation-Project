import torch
import numpy as np
import torch.nn as nn
import copy
from collections import Counter
import os
import random


def corrcoef(x, rowvar=True):
    """
    code from
    https://github.com/AllenCellModeling/pytorch_integrated_cell/blob/8a83fc6f8dc79037f4b681d9d7ef0bc5b91e9948/integrated_cell/corr_stats.py
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor
    rowvar : bool, default True means every single row is a variable, and every single column is an observation, e.g. a sample
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    # 计算每个变量的均值，默认每行是一个变量，每列是一个sample
    if not rowvar and len(x.size()) != 1:
        x = x.T
    mean_x = torch.mean(x, 1).unsqueeze(1)
    # xm(j, i)是第i个sample的第j个变量，已经被减去了j变量的均值，等于论文中的F(si)j- uj,
    # xm(k, i)是第i个sample的第k个变量，已经被减去了k变量的均值，等于论文中的F(si)k- uk,
    xm = x.sub(mean_x.expand_as(x))
    # c(j, k) 等于论文中 M(j, k)的分子, c也是F(s)的协方差矩阵Cov(F(s), F(s))
    c = xm.mm(xm.t())
    # 协方差矩阵一般会除以 num_sample - 1
    # c = c / (x.size(1) - 1)

    # normalize covariance matrix
    # dj是每个变量的方差, E[(F(s)j - uj)^2]，也即j == k 时的分子
    d = torch.diag(c)
    # 取标准差
    stddev = torch.pow(d + 1e-7, 0.5)  # 防止出现0，导致nan
    # 论文中除以的分母
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def disentangling_loss(feature):
    # feature = [batch_size, hid_dim]
    M = corrcoef(feature, rowvar=False)
    # M = [hid_dim, hid_dim]
    loss_decorr = 0.5 * (torch.sum(torch.pow(M, 2)) - torch.sum(torch.pow(torch.diag(M), 2)))
    return loss_decorr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

    elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    else:
        for param in module.parameters():
            nn.init.uniform_(param, -0.02, 0.02)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def print_metrics(metrics, mode=''):
    # TODO 把print metrics统一起来
    print("#" * 20, mode + ' metrics ', "#" * 20)
    for k, v in metrics.items():
        print(f'\t{k}: {v:.5f}')
    #  print("#" * 20, ' end ', "#" * (24 + len(mode)))


def write_metrics(metrics, file, mode=''):
    file.write("#" * 20 + mode + ' metrics ' + "#" * 20 + '\n')
    for k, v in metrics.items():
        file.write(f'\t{k}: {v:.5f}\n')
    file.write("#" * 20 + ' end ' + "#" * 20 + '\n')


def write_metrics_to_writer(metrics, writer, global_step, mode=''):
    for k, v in metrics.items():
        writer.add_scalar(f'{mode}_{k}', v, global_step)


def clones(module, N):
    """生成N个同样的layers"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """将后续部分mask掉"""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Distinct:
    def __init__(self, n=2, exclude_tokens=None):
        self.n = n
        self.exclude_tokens = exclude_tokens
        if self.exclude_tokens:
            self.exclude_tokens = set(self.exclude_tokens)
        self.n_grams_all = [Counter() for _ in range(n)]

    def forward(self, seqs):
        for seq in seqs:
            if self.exclude_tokens:
                seq = list(filter(lambda x: x not in self.exclude_tokens, seq))
            for i in range(self.n):
                k_grams = Counter(zip(*[seq[k:] for k in range(i + 1)]))
                self.n_grams_all[i].update(k_grams)

    def get_metric(self, reset=False):
        dist_n = dict()
        for i, dist in enumerate(self.n_grams_all):
            dist_n[f'dist_{i + 1}'] = (len(dist) + 1e-12) / (sum(dist.values()) + 1e-5)
        if reset:
            self.n_grams_all = [Counter() for _ in range(self.n)]
        return dist_n


from allennlp.training.metrics import BLEU as AllennlpBLEU
class BLEU:
    """Simple wrapper of allennlp.training.metrics.BLEU, compute bleu1~n where n is defined by user,
    n should be in range [1, 4]"""
    def __init__(self, n, exclude_indices=None):
        self.weights = [[1, 0, 0, 0],
                        [0.5, 0.5, 0, 0],
                        [1/3, 1/3, 1/3, 0],
                        [0.25, 0.25, 0.25, 0.25]]
        self.n = n
        self.bleus = [AllennlpBLEU(ngram_weights=self.weights[i], exclude_indices=exclude_indices) for i in range(n)]

    def forward(self, predictions, gold_targets):
        for bleu in self.bleus:
            bleu(predictions, gold_targets)

    def get_metric(self, reset=False):
        metrics = dict()
        for i, bleu in enumerate(self.bleus):
            metrics[f"bleu_{i + 1}"] = bleu.get_metric(reset=reset)['BLEU']
        return metrics


class LabelSmoothing(nn.Module):
    "Implement label smoothing using KL div loss"
    def __init__(self, args, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        assert smoothing > 0 and smoothing < 1
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.criterion.to(args.device)
        # ignore_index只是让padding_idx的梯度不参与计算，并没有把padding_idx上的loss排除在总的loss之外，
        # 所以这样计算得到的loss是偏大的，应该再设置weight为[0,1,1,…]（0表示padding_idx），这样就能保证总的loss不包括padding_idx上的loss
        weight = torch.ones(size)
        weight[padding_idx] = 0
        weight.to(args.device)
        self.ce_criterion = nn.NLLLoss(reduction='sum', ignore_index=padding_idx, weight=weight)
        self.ce_criterion.to(args.device)
        
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # x = [batch, category]
        # target = [batch, ]
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        #  先把true_dist的元素全填为smoothing / (size - 2) 的值，-2是因为真实标签位置和padding位置的概率都要另设
        true_dist.fill_(self.smoothing / (self.size - 2))
        #  再把true_dist在列上以target为索引的地方的值变为confidence
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        #  把padding的位置概率都变为0
        true_dist[:, self.padding_idx] = 0
        #  把target就预测padding token的整个概率分布都设为0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        #  用true_dist这个经过平滑的概率分布去替代target的one-hot概率分布是为了避免模型的预测分布也向one-hot靠近
        #  避免模型变得太过confident，模型学着预测时变得更不确定，这对ppl有伤害，但是能够提升预测的准确性和BLEU分数
        #  当x的概率分布很尖锐时，loss将变大
        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False)), self.ce_criterion(x, target.long())


def write_src_tgt_to_file(args, batch, src_vocab, tgt_vocab, mode='valid'):
    num_lines = 0
    with open(os.path.join(args.save_dir, 'posts-' + mode + '.txt'), 'a') as fp:
        with open(os.path.join(args.save_dir, 'answers-'+ mode + '.txt'), 'a') as fa:
            src, src_len = batch.src
            src = src.T
            # tgt = [tgt len, batch size]
            tgt, tgt_len = batch.tgt
            tgt = tgt.T
            src = [[src_vocab.itos[x] for x in ex] for ex in src]
            tgt = [[tgt_vocab.itos[x] for x in ex] for ex in tgt]
            for k in range(len(src)):
                fp.write(' '.join(src[k][:src_len[k]]).strip() + '\n')
                fa.write(' '.join(tgt[k][1:tgt_len[k] - 1]).strip() + '\n')
                num_lines += 1
    return num_lines


def write_pred_to_file(args, response, tgt_vocab, mode='valid'):
    num_lines = 0
    with open(os.path.join(args.save_dir, 'responses-' + mode + '.txt'), 'a') as fr:
        #  response = [batch_size, len]
        res = []
        TGT_EOS_IDX = tgt_vocab.stoi['<eos>']

        for ex in response:
            wrote = False
            cur = []
            for x in ex:
                if x != TGT_EOS_IDX:
                    cur.append(tgt_vocab.itos[x])
                else:
                    fr.write(' '.join(cur).strip() + '\n')
                    wrote = True
                    num_lines += 1
                    break
            if not wrote:
                fr.write(' '.join(cur).strip() + '\n')
                num_lines += 1

    return num_lines


def setup_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

