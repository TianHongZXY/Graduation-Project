from tqdm import tqdm
import torch
import math
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import write_src_tgt_to_file, write_pred_to_file
import numpy as np
import torch.autograd as autograd


def train(args, model, iterator, optimizer, criterion, fields, writer=None, pmi=None):
    model.train()

    start = time.time()
    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    vocab_size = len(fields['tgt'].vocab)
    # TODO 为v5暂时修改
    #  kl_criterion = nn.KLDivLoss()

    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        batch_size = src.size(1)
        seq_len = src.size(0)
        if pmi is not None:
            # src_idx = [batch size * seq len, ]
            src_idx = src.cpu().numpy().T.reshape(-1)
            src_pmi = pmi[src_idx, :]
            # src_pmi = [batch size * seq len, tgt vocab size]
            src_pmi = torch.FloatTensor(src_pmi.todense()).to(args.device)
            src_pmi = src_pmi.view(batch_size, seq_len, -1)
            # src_pmi = [batch size, tgt vocab size]
            src_pmi = torch.sum(src_pmi, dim=1)
        # tgt = [tgt len, batch size]
        tgt, tgt_len = batch.tgt
        tgt_clone = tgt.clone()
        ntokens = (tgt[1:] != tgt_padding_idx).data.sum()
        total_tokens += ntokens
        optimizer.zero_grad()

        if args.model == 'pmi_seq2seq':
            # output = [tgt len, batch size, vocab_size]
            output = model(src, src_len.cpu().long(), tgt, pmi=src_pmi, teacher_forcing_ratio=args.teaching_rate)
        elif args.model == 'seq2seq':
            output = model(src, src_len.cpu().long(), tgt, teacher_forcing_ratio=args.teaching_rate)

        # 用于求KL-DIV Loss或NLL Loss需要先求log softmax
        # TODO 为v5临时修改
        logits = output
        #  logits, kl_logits = output
        #  entropy = - F.softmax(logits[1:], dim=-1) * F.log_softmax(logits[1:], dim=-1)
        #  entropy = 0.0001 * torch.sum(entropy)
        #  kl_loss = 100 * kl_criterion(F.log_softmax(kl_logits, dim=-1), F.softmax(src_pmi, dim=-1))
        output = F.log_softmax(logits, dim=-1)

        #  output = model.forward_parallel(src, src_len.cpu().long(), tgt)
        # output = [(tgt len - 1) * batch size, vocab_size]
        # tgt = [(tgt len - 1) * batch size]
        output = output[1:].reshape(-1, vocab_size)
        tgt = tgt[1:].reshape(-1)

        if args.smoothing > 0:
            loss, ce_loss = criterion(output, tgt)
            loss /= ntokens
            ce_loss /= ntokens
        else:
            loss = criterion(output, tgt)
            loss /= ntokens
            ce_loss = loss
        # TODO 为v5临时修改
        #  loss = loss + kl_loss - entropy
        # TODO 为梯度惩罚临时修改
        #  autograd.backward(loss, create_graph=True)
        #############grad penalty################
        x_grad = autograd.grad(loss, model.src_embed.lut.weight, create_graph=True)[0]
        gp = torch.pow(torch.norm(x_grad), 2)
        loss += gp
        loss.backward()
        #  update1 = model.src_embed.lut.weight.grad.data.clone()
        #  gp.backward()
        #  update2 = model.src_embed.lut.weight.grad.data.clone()
        #  print('update1: ', update1)
        #  print('update2: ', update2)
        #  diff_norm = (update1-update2).norm().item()
        #  print(diff_norm)
        #########################################
        
        ##############grad attack################
        #  grad_attack = model.src_embed.lut.weight.grad
        #  model.src_embed.lut.weight.data.add_(grad_attack)
        #  if args.model == 'pmi_seq2seq':
        #      output = model(src, src_len.cpu().long(), tgt_clone, pmi=src_pmi, teacher_forcing_ratio=args.teaching_rate)
        #  elif args.model == 'seq2seq':
        #      output = model(src, src_len.cpu().long(), tgt_clone, teacher_forcing_ratio=args.teaching_rate)
        #
        #  logits = output
        #  output = F.log_softmax(logits, dim=-1)
        #  output = output[1:].reshape(-1, vocab_size)
        #  if args.smoothing > 0:
        #      loss_adv, _ = criterion(output, tgt)
        #      loss_adv /= ntokens
        #  else:
        #      loss_adv = criterion(output, tgt)
        #      loss_adv /= ntokens
        #  loss_adv.backward()
        ########################################
        optimizer.step()
        i += 1
        if i % 50 == 0:
            elapsed = time.time() - start
            # TODO 为v5临时修改
            #  print("Global Step: %d\tEpoch Step: %d\tLoss: %f\tTokens per Sec: %f\tlr: %f\tkl_loss: %f\tentropy: %f" %
            #              (optimizer.get_global_step(), i, ce_loss.item(), total_tokens / elapsed, optimizer.rate(), kl_loss, entropy))
            print("Global Step: %d\tEpoch Step: %d\tLoss: %f\tTokens per Sec: %f\tlr: %f\tgp %f" %
                        (optimizer.get_global_step(), i, ce_loss.item(), total_tokens / elapsed, optimizer.rate(), gp.item()))
        if writer:
            writer.add_scalar('Train_Loss', ce_loss.item(), optimizer.get_global_step())
            writer.add_scalar('Train_PPL', math.exp(ce_loss.item()), optimizer.get_global_step())

        total_loss += ce_loss.item() * ntokens
        #  break

    return {
            'epoch_loss': total_loss / total_tokens,
            'PPL': math.exp(total_loss / total_tokens),
            }


def evaluate(args, model, iterator, criterion, fields, pmi=None):
    model.eval()

    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    vocab_size = len(fields['tgt'].vocab)

    with torch.no_grad():
        for batch in tqdm(iterator):
            src, src_len = batch.src
            batch_size = src.size(1)
            seq_len = src.size(0)
            if pmi is not None:
                # src_idx = [batch size * seq len, ]
                src_idx = src.cpu().numpy().T.reshape(-1)
                src_pmi = pmi[src_idx, :]
                # src_pmi = [batch size * seq len, tgt vocab size]
                src_pmi = torch.FloatTensor(src_pmi.todense()).to(args.device)
                src_pmi = src_pmi.view(batch_size, seq_len, -1)
                # src_pmi = [batch size, tgt vocab size]
                src_pmi = torch.sum(src_pmi, dim=1)
            # tgt = [tgt len, batch size]
            tgt, tgt_len = batch.tgt
            ntokens = (tgt[1:] != tgt_padding_idx).data.sum()
            total_tokens += ntokens

            if args.model == 'pmi_seq2seq':
                # output = [tgt len, batch size, vocab_size]
                output = model(src, src_len.cpu().long(), tgt, pmi=src_pmi, teacher_forcing_ratio=1)
            elif args.model == 'seq2seq':
                output = model(src, src_len.cpu().long(), tgt, teacher_forcing_ratio=1)

            # pred = [batch size, tgt len - 1]
            #  pred = output[1:].argmax(-1).T

            # output = [(tgt len - 1) * batch size, vocab_size]
            # tgt = [(tgt len - 1) * batch size]
            # TODO 为v5临时修改
            #  output, _ = output
            output = output[1:].reshape(-1, vocab_size)
            tgt = tgt[1:].reshape(-1)

            # 用于求KL-DIV Loss或NLL Loss需要先求log softmax
            output = F.log_softmax(output, dim=-1)
            if args.smoothing > 0:
                _, ce_loss = criterion(output, tgt)
            else:
                ce_loss = criterion(output, tgt)
            total_loss += ce_loss.item()
            #  break

    metrics = {
            'epoch_loss': total_loss / total_tokens,
            'PPL': math.exp(total_loss / total_tokens),
            }

    return metrics


def inference(args, model, iterator, fields, mode, pmi=None):
    model.eval()
    src_vocab = fields['src'].vocab
    tgt_vocab = fields['tgt'].vocab

    with torch.no_grad():
        for batch in tqdm(iterator):
            src_num_lines = write_src_tgt_to_file(args, batch, src_vocab, tgt_vocab, mode)
            src, src_len = batch.src
            batch_size = src.size(1)
            seq_len = src.size(0)
            if pmi is not None:
                # src_idx = [batch size * seq len, ]
                src_idx = src.cpu().numpy().T.reshape(-1)
                src_pmi = pmi[src_idx, :]
                # src_pmi = [batch size * seq len, tgt vocab size]
                src_pmi = torch.FloatTensor(src_pmi.todense()).to(args.device)
                src_pmi = src_pmi.view(batch_size, seq_len, -1)
                # src_pmi = [batch size, tgt vocab size]
                src_pmi = torch.sum(src_pmi, dim=1)
            # tgt = [tgt len, batch size]
            tgt, tgt_len = batch.tgt
            # output = [tgt len, batch size, vocab_size]
            if args.model == 'pmi_seq2seq':
                output = model(src, src_len.cpu().long(), tgt, pmi=src_pmi, teacher_forcing_ratio=0)
            elif args.model == 'seq2seq':
                output = model(src, src_len.cpu().long(), tgt, teacher_forcing_ratio=0)

            # TODO 为v5临时修改
            #  output, _ = output

            # pred = [batch size, tgt len - 1]
            pred = output[1:].argmax(-1).T
            num_lines = write_pred_to_file(args, pred, fields['tgt'].vocab, mode)
            try:
                assert src_num_lines == num_lines
            except AssertionError:
                print('src num line', src_num_lines)
                print('pred num line', num_lines)
                print('src', src.size(), src)
                print('pred', pred.size(), pred)

