from tqdm import tqdm
import torch
import math
import time


def train(args, model, iterator, optimizer, criterion, fields, writer=None):
    model.train()

    start = time.time()
    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    for i, batch in enumerate(iterator):
        src, src_len = batch.src
        # tgt = [tgt len, batch size]
        tgt, tgt_len = batch.tgt
        ntokens = (tgt[1:] != tgt_padding_idx).data.sum()
        total_tokens += ntokens
        optimizer.zero_grad()

        # output = [tgt len, batch size, vocab_size]
        output = model(src, src_len.cpu().long(), tgt, teacher_forcing_ratio=args.teaching_rate)
        #  output = model.forward_parallel(src, src_len.cpu().long(), tgt)
        vocab_size = len(fields['tgt'].vocab)
        # output = [(tgt len - 1) * batch size, vocab_size]
        # tgt = [(tgt len - 1) * batch size]
        output = output[1:].reshape(-1, vocab_size)
        tgt = tgt[1:].reshape(-1)

        loss = criterion(output, tgt)
        loss /= ntokens
        loss.backward()

        optimizer.step()
        i += 1
        if i % 50 == 0:
            elapsed = time.time() - start
            print("Global Step: %d\tEpoch Step: %d\tLoss: %f\tTokens per Sec: %f\tlr: %f" %
                        (optimizer.get_global_step(), i, loss, total_tokens / elapsed, optimizer.rate()))
        if writer:
            writer.add_scalar('Train_Loss', loss.item(), optimizer.get_global_step())
            writer.add_scalar('Train_PPL', math.exp(loss.item()), optimizer.get_global_step())

        total_loss += loss.item() * ntokens
        #  break
    return {'epoch_loss': total_loss / total_tokens,
            'PPL': math.exp(total_loss / total_tokens),
            }


def evaluate(args, model, iterator, criterion, fields, bleu=None, dist=None):
    model.eval()

    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]

    with torch.no_grad():
        for batch in tqdm(iterator):
            src, src_len = batch.src
            # tgt = [tgt len, batch size]
            tgt, tgt_len = batch.tgt
            ntokens = (tgt[1:] != tgt_padding_idx).data.sum()
            total_tokens += ntokens
            vocab_size = len(fields['tgt'].vocab)
            # output = [tgt len, batch size, vocab_size]
            output = model(src, src_len.cpu().long(), tgt, 0)  # turn off teacher forcing
            # pred = [batch size, tgt len - 1]
            pred = output[1:].argmax(-1).T
            if bleu:
                gold = tgt[1:].T
                bleu(predictions=pred, gold_targets=gold)
            if dist:
                dist.forward([hyps for hyps in pred.detach().cpu().numpy()])

            # output = [(tgt len - 1) * batch size, vocab_size]
            # tgt = [(tgt len - 1) * batch size]
            output = output[1:].reshape(-1, vocab_size)
            tgt = tgt[1:].reshape(-1)

            loss = criterion(output, tgt)
            total_loss += loss.item()
            #  break
    metrics = {'epoch_loss': total_loss / total_tokens,
               'PPL': math.exp(total_loss / total_tokens),
               }
    if bleu:
        metrics['bleu'] = bleu.get_metric(reset=True)['BLEU']
    if dist:
        dist_n = dist.get_metric(reset=True)
        metrics.update(dist_n)

    return metrics
