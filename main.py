from model.Encoder import RNNBaseEncoder
from model.Decoder import RNNBaseDecoder, Generator
from model.EncoderDecoder import RNNBaseSeq2Seq
from model.Embedding import Embedding
from data.dataset import seq2seq_dataset
from train import train, evaluate
import torch.optim as optim
import time
import math
from tqdm import tqdm
from allennlp.training.metrics import BLEU
from optim.Optim import NoamOptimWrapper
from utils import *
import argparse
import os
from datetime import datetime
from dateutil import tz, zoneinfo
import json


def main():
    parser = argparse.ArgumentParser(
        description='allennlp seq2seq'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--save', action="store_true", help='whether to save model or not')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--emb_dim', default=300, type=int, help='embedding dim')
    parser.add_argument('--hid_dim', default=300, type=int, help='hidden dim of lstm')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio')
    parser.add_argument('--n_epochs', default=30, type=int, help='num of train epoch')
    parser.add_argument('--min_freq', default=1, type=int, help='minimal occur times for vocabulary')
    parser.add_argument('--clip', default=None, type=float, help='grad clip')
    parser.add_argument('--maxlen', default=None, type=int, help='max length of text')
    parser.add_argument('--dataset_dir_path', default=None, type=str, help='path to directory where data file is saved')
    parser.add_argument('--train_file', default=None, type=str, help='train file name')
    parser.add_argument('--valid_file', default=None, type=str, help='valid file name')
    parser.add_argument('--test_file', default=None, type=str, help='test file name')
    parser.add_argument('--save_dir', default='models', type=str, help='save dir')
    parser.add_argument('--vocab_file', default=None, type=str, help='predefined vocab file')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--teaching_rate', default=0.5, type=float, help='teaching_rate is probability to use teacher forcing')
    parser.add_argument('--pretrained_embed_file', default=None, type=str, help='torchtext vector name')
    parser.add_argument('--warmup', default=0, type=int, help='warmup steps, 0 means not using NoamOpt')
    parser.add_argument('--cell_type', default='LSTM', type=str, help='cell type of encoder/decoder, LSTM or GRU')


    args, unparsed = parser.parse_known_args()
    if args.save:
        tz_sh = tz.gettz('Asia/Shanghai')
        save_dir = os.path.join(args.save_dir, 'run' + str(datetime.now(tz=tz_sh)).replace(":", "-").split(".")[0].replace(" ", '.'))
        args.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    args.device = device

    dataset = seq2seq_dataset(args)
    SRC = dataset['fields']['src']
    TGT = dataset['fields']['tgt']
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim
    N_LAYERS = 1
    ENC_DROPOUT = args.dropout
    DEC_DROPOUT = args.dropout
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TGT_PAD_IDX = TGT.vocab.stoi[TGT.pad_token]
    N_EPOCHS = args.n_epochs
    CLIP = args.clip

    src_embedding = Embedding(len(SRC.vocab), EMB_DIM, padding_idx=SRC_PAD_IDX, dropout=ENC_DROPOUT)
    tgt_embedding = Embedding(len(TGT.vocab), EMB_DIM, padding_idx=TGT_PAD_IDX, dropout=DEC_DROPOUT)
    if args.pretrained_embed_file:
        # 权重在词汇表vocab的vectors属性中
        src_pretrained_vectors = SRC.vocab.vectors
        tgt_pretrained_vectors = TGT.vocab.vectors
        # 指定嵌入矩阵的初始权重
        src_embedding.lut.weight.data.copy_(src_pretrained_vectors)
        tgt_embedding.lut.weight.data.copy_(tgt_pretrained_vectors)
        print("pretrained vectors loaded successfully!")
    enc = RNNBaseEncoder(args.cell_type, EMB_DIM, HID_DIM, N_LAYERS, bidirectional=False, dropout=ENC_DROPOUT)
    dec = RNNBaseDecoder(args.cell_type, EMB_DIM, HID_DIM, N_LAYERS, dropout=DEC_DROPOUT)
    generator = Generator(HID_DIM, len(TGT.vocab))
    model = RNNBaseSeq2Seq(enc, dec, src_embedding, tgt_embedding, generator).to(device)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.warmup > 0:
        optimizer = NoamOptimWrapper(args.hid_dim, 1, args.warmup, optimizer)
    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX, reduction='sum')

    best_epoch = 0
    best_valid_loss = float('inf')
    best_bleu = 0
    best_dist_1 = 0
    best_dist_2 = 0
    global_step = 0
    bleu = BLEU(exclude_indices={TGT_PAD_IDX, TGT.vocab.stoi[TGT.eos_token], TGT.vocab.stoi[TGT.init_token]})
    dist = Distinct(exclude_tokens={TGT_PAD_IDX, TGT.vocab.stoi[TGT.eos_token], TGT.vocab.stoi[TGT.init_token]})

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_metrics = train(args, model, dataset['train_iterator'], optimizer, criterion, fields=dataset['fields'])
        valid_metrics = evaluate(args, model, dataset['valid_iterator'], criterion, bleu=bleu, fields=dataset['fields'], dist=dist)
        test_metrics = evaluate(args, model, dataset['test_iterator'], criterion, bleu=bleu, fields=dataset['fields'], dist=dist)
        end_time = time.time()
        global_step += len(dataset['train_iterator'])
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Global step: {global_step} | Time: {epoch_mins}m {epoch_secs}s')
        for metrics, mode in zip([train_metrics, valid_metrics, test_metrics], ['Train', 'Valid', 'Test']):
            print_metrics(metrics, mode=mode)

        # TODO 优化存储logfile,取消hard-code模式
        if args.save:
            best_valid_loss = valid_metrics['epoch_loss']  if valid_metrics['epoch_loss'] < best_valid_loss else best_valid_loss
            best_epoch = epoch if valid_metrics['epoch_loss'] == best_valid_loss else best_epoch
            best_bleu = valid_metrics['bleu'] if valid_metrics['bleu'] > best_bleu else best_bleu
            best_dist_1 = valid_metrics['dist_1'] if valid_metrics['dist_1'] > best_dist_1 else best_dist_1
            best_dist_2 = valid_metrics['dist_2'] if valid_metrics['dist_2'] > best_dist_2 else best_dist_2
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_global_step-{global_step}.pt'))
            with open(os.path.join(save_dir, f'log_epoch-{epoch+1}_global_step-{global_step}.txt'), 'w') as log_file:
                valid_metrics['Best Epoch'] = best_epoch + 1
                valid_metrics['Best Valid Loss'] = best_valid_loss
                valid_metrics['Best PPL'] = math.exp(best_valid_loss)
                valid_metrics['Best BLEU'] = best_bleu
                valid_metrics['Best Dist-1'] = best_dist_1
                valid_metrics['Best Dist-2'] = best_dist_2

                for metric, performance in valid_metrics.items():
                    log_file.write(f'{metric}: {performance}\n')
                #  for metrics, mode in zip([train_metrics, valid_metrics, test_metrics], ['Train', 'Valid', 'Test']):
                #      write_metrics(metrics, log_file, mode=mode)


if __name__ == '__main__':
    main()
