from tqdm import tqdm
import torch
import math


def train(model, iterator, optimizer, criterion, clip, fields):
    model.train()

    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]
    # i = 0
    for batch in tqdm(iterator):
        src, src_len = batch.src
        # tgt = [batch size, tgt len]
        tgt, tgt_len = batch.tgt
        ntokens = (tgt[:, 1:] != tgt_padding_idx).data.sum()
        total_tokens += ntokens
        optimizer.zero_grad()

        # output = model(src, src_len.cpu().long(), tgt[:, :-1], teacher_forcing_ratio=1)
        output = model.forward_parallel(src, src_len.cpu().long(), tgt[:, :-1])
        # output = [batch size, tgt len - 1, vocab_size]
        vocab_size = len(fields['tgt'].vocab)
        output = output.reshape(-1, vocab_size)
        tgt = tgt[:, 1:].reshape(-1)

        # tgt = [batch size * (tgt len - 1),]
        # output = [batch size * (tgt len - 1) , vocab_size]
        loss = criterion(output.float(), tgt)

        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        total_loss += loss.item()
        # i += 1
        # if i == 10:
        #     break
    return {'epoch_loss': total_loss / total_tokens,
            'PPL': math.exp(total_loss / total_tokens),
            }


def evaluate(model, iterator, criterion, fields, bleu=None):
    model.eval()

    total_loss = 0
    total_tokens = 0
    tgt_padding_idx = fields['tgt'].vocab.stoi[fields['tgt'].pad_token]

    with torch.no_grad():
        for batch in tqdm(iterator):
            src, src_len = batch.src
            # tgt = [batch size, tgt len]
            tgt, tgt_len = batch.tgt
            ntokens = (tgt[:, 1:] != tgt_padding_idx).data.sum()
            total_tokens += ntokens
            vocab_size = len(fields['tgt'].vocab)
            # output = [batch size, tgt len - 1, vocab_size]
            output = model(src, src_len.cpu().long(), tgt[:, :-1], 0)  # turn off teacher forcing

            if bleu:
                gold = tgt[:, 1:]
                pred = output.argmax(-1)
                bleu(predictions=pred, gold_targets=gold)

            # output = [batch size * (tgt len - 1) , vocab_size]
            output = output.reshape(-1, vocab_size)
            # tgt = [batch size * (tgt len - 1)]
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output.float(), tgt)
            total_loss += loss.item()
    metrics = {'epoch_loss': total_loss / total_tokens,
               'PPL': math.exp(total_loss / total_tokens),
               }
    if bleu:
        metrics['bleu'] = bleu.get_metric(reset=True)['BLEU']

    return metrics
