from tqdm import tqdm
import torch
import math
import time
import torch.nn.functional as F
from utils import write_src_tgt_to_file, write_pred_to_file


def train(args, model, iterator, optimizer, criterion, fields, writer=None):
    model.train()

    start = time.time()
    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    vocab_size = len(fields['tgt'].vocab)

    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        # tgt = [tgt len, batch size]
        tgt, tgt_len = batch.tgt
        ntokens = (tgt[1:] != tgt_padding_idx).data.sum()
        total_tokens += ntokens
        optimizer.zero_grad()

        # output = [tgt len, batch size, vocab_size]
        output = model(src, src_len.cpu().long(), tgt, teacher_forcing_ratio=args.teaching_rate)
        # 用于求KL-DIV Loss或NLL Loss需要先求log softmax
        output = F.log_softmax(output, dim=-1)

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
        loss.backward()

        optimizer.step()
        i += 1
        if i % 50 == 0:
            elapsed = time.time() - start
            print("Global Step: %d\tEpoch Step: %d\tLoss: %f\tTokens per Sec: %f\tlr: %f" %
                        (optimizer.get_global_step(), i, ce_loss.item(), total_tokens / elapsed, optimizer.rate()))
        if writer:
            writer.add_scalar('Train_Loss', ce_loss.item(), optimizer.get_global_step())
            writer.add_scalar('Train_PPL', math.exp(ce_loss.item()), optimizer.get_global_step())

        total_loss += ce_loss.item() * ntokens
        #  break

    return {
            'epoch_loss': total_loss / total_tokens,
            'PPL': math.exp(total_loss / total_tokens),
            }


def evaluate(args, model, iterator, criterion, fields):
    model.eval()

    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    vocab_size = len(fields['tgt'].vocab)

    with torch.no_grad():
        for batch in tqdm(iterator):
            src, src_len = batch.src
            # tgt = [tgt len, batch size]
            tgt, tgt_len = batch.tgt
            ntokens = (tgt[1:] != tgt_padding_idx).data.sum()
            total_tokens += ntokens
            # output = [tgt len, batch size, vocab_size]
            output = model(src, src_len.cpu().long(), tgt, 0)  # turn off teacher forcing

            # pred = [batch size, tgt len - 1]
            #  pred = output[1:].argmax(-1).T

            # output = [(tgt len - 1) * batch size, vocab_size]
            # tgt = [(tgt len - 1) * batch size]
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


def inference(args, model, iterator, criterion, fields, mode):
    model.eval()
    src_vocab = fields['src'].vocab
    tgt_vocab = fields['tgt'].vocab

    with torch.no_grad():
        for batch in tqdm(iterator):
            src_num_lines = write_src_tgt_to_file(args, batch, src_vocab, tgt_vocab, mode)
            src, src_len = batch.src
            # tgt = [tgt len, batch size]
            tgt, tgt_len = batch.tgt
            # output = [tgt len, batch size, vocab_size]
            output = model(src, src_len.cpu().long(), tgt, 0)  # turn off teacher forcing

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

