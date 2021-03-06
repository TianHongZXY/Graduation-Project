from model.Encoder import RNNBaseEncoder
from model.Decoder import RNNBaseDecoder, Generator, LuongAttnRNNDecoder
from model.EncoderDecoder import RNNBaseSeq2Seq, RNNBasePMISeq2Seq_v4 as RNNBasePMISeq2Seq
from model.Embedding import Embedding
from model.modules import LayerNorm
from data.dataset import seq2seq_dataset, load_iwslt
from train import train, evaluate, inference
import torch.optim as optim
import time
import math
from tqdm import tqdm
#  from allennlp.training.metrics import BLEU
from optim.Optim import NoamOptimWrapper, AdamOptimizer
from utils import *
import argparse
import os
from datetime import datetime
from dateutil import tz, zoneinfo
import json
from torch.utils.tensorboard import SummaryWriter
from transformers.models.bert.tokenization_bert import BertTokenizer
import logging
import coloredlogs
from nlp_metrics import calc_metrics
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def main():
    parser = argparse.ArgumentParser(
        description='seq2seq'
    )

    parser.add_argument('--model', default='seq2seq', type=str, help='which model you are going to train, now including [seq2seq, pmi_seq2seq]')
    parser.add_argument('--attn', default=None, type=str, help='which attention method to use, including [dot, general, concat]')
    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--save', action="store_true", help='whether to save model or not')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--emb_dim', default=300, type=int, help='embedding dim')
    parser.add_argument('--enc_hid_dim', default=300, type=int, help='hidden dim of lstm')
    parser.add_argument('--dec_hid_dim', default=300, type=int, help='hidden dim of lstm')
    parser.add_argument('--birnn', action='store_true', help='whether to use bidirectional rnn, default False')
    parser.add_argument('--n_layers', default=1, type=int, help='layer num of encoder and decoder')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio')
    parser.add_argument('--n_epochs', default=30, type=int, help='num of train epoch')
    parser.add_argument('--min_freq', default=1, type=int, help='minimal occur times for vocabulary')
    parser.add_argument('--clip', default=None, type=float, help='grad clip')
    parser.add_argument('--maxlen', default=None, type=int, help='max length of text')
    parser.add_argument('--dataset_dir_path', default=None, type=str, help='path to directory where data file is saved')
    parser.add_argument('--tokenizer', default='spacy_en', type=str, help='which tokenizer to use for the dataset')
    parser.add_argument('--train_file', default=None, type=str, help='train file name')
    parser.add_argument('--valid_file', default=None, type=str, help='valid file name')
    parser.add_argument('--test_file', default=None, type=str, help='test file name')
    parser.add_argument('--save_dir', default='models', type=str, help='save dir')
    parser.add_argument('--vocab_file', default=None, type=str, help='predefined vocab file')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--teaching_rate', default=1, type=float, help='teaching_rate is probability to use teacher forcing')
    parser.add_argument('--pretrained_embed_file', default=None, type=str, help='torchtext vector name')
    parser.add_argument('--warmup', default=0, type=int, help='warmup steps, 0 means not using NoamOpt')
    parser.add_argument('--cell_type', default='LSTM', type=str, help='cell type of encoder/decoder, LSTM or GRU')
    parser.add_argument('--comment', default='', type=str, help='comment, will be used as prefix of save directory')
    parser.add_argument('--smoothing', default=0.0, type=float, help='smoothing rate of computing kl div loss')
    parser.add_argument('--max_vocab_size', default=None, type=int, help='max size of vocab')
    parser.add_argument('--serialize', action='store_true', help='whether to serialize examples and vocab')
    parser.add_argument('--use_serialized', action='store_true', help='whether to use serialized dataset')
    parser.add_argument('--model_path', default=None, type=str, help='restore model to continue training')
    parser.add_argument('--global_step', default=0, type=int, help='global step for continuing training')
    parser.add_argument('--inference', action='store_true', help='inference mode')
    parser.add_argument('--seed', default=20020206, type=int, help='random seed')
    parser.add_argument('--ln', action='store_true', help='whether to use layernorm, if model is pmi_seq2seq, use conditional layernorm as default')
    parser.add_argument('--patience', default=None, type=int, help="stop when {patience} continued epochs giving  no improved performance")

    args, unparsed = parser.parse_known_args()
    setup_random_seed(args.seed)
    writer = None
    if args.save:
        tz_sh = tz.gettz('Asia/Shanghai')
        save_dir = os.path.join(args.save_dir, args.comment + 'run' + str(datetime.now(tz=tz_sh)).replace(":", "-").split(".")[0].replace(" ", '.'))
        if args.model_path:
            save_dir = os.path.split(args.model_path)[0]
        args.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        writer = SummaryWriter(os.path.join(save_dir, 'summary'))

    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')
    args.device = device

    if args.tokenizer == 'spacy_en':
        dataset = seq2seq_dataset(args)
    elif args.tokenizer == 'jieba':
        from data.dataset import jieba_tokenize
        dataset = seq2seq_dataset(args, tokenizer=jieba_tokenize)
    elif args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset = seq2seq_dataset(args, tokenizer=tokenizer.tokenize)
    elif args.tokenizer == 'whitespace':
        from data.dataset import whitespace_tokenize
        dataset = seq2seq_dataset(args, tokenizer=whitespace_tokenize)

    #  dataset = load_iwslt(args)

    SRC = dataset['fields']['src']
    TGT = dataset['fields']['tgt']

    EMB_DIM = args.emb_dim
    ENC_HID_DIM = args.enc_hid_dim
    DEC_HID_DIM = args.dec_hid_dim
    N_LAYERS = args.n_layers
    ENC_DROPOUT = args.dropout
    DEC_DROPOUT = args.dropout
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TGT_PAD_IDX = TGT.vocab.stoi[TGT.pad_token]
    N_EPOCHS = args.n_epochs
    CLIP = args.clip

    src_embedding = Embedding(len(SRC.vocab), EMB_DIM, padding_idx=SRC_PAD_IDX, dropout=ENC_DROPOUT)
    tgt_embedding = Embedding(len(TGT.vocab), EMB_DIM, padding_idx=TGT_PAD_IDX, dropout=DEC_DROPOUT)
    if args.pretrained_embed_file:
        # ??????????????????vocab???vectors?????????
        src_pretrained_vectors = SRC.vocab.vectors
        tgt_pretrained_vectors = TGT.vocab.vectors
        # ?????????????????????????????????
        src_embedding.lut.weight.data.copy_(src_pretrained_vectors)
        tgt_embedding.lut.weight.data.copy_(tgt_pretrained_vectors)
        print("pretrained vectors loaded successfully!")

    enc = RNNBaseEncoder(args.cell_type, EMB_DIM, ENC_HID_DIM, N_LAYERS, bidirectional=args.birnn, dropout=ENC_DROPOUT, layernorm=args.ln)
    if args.attn is not None:
        dec = LuongAttnRNNDecoder(args.cell_type, EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attn_method=args.attn, num_layers=N_LAYERS, dropout=DEC_DROPOUT)
    else:
        dec = RNNBaseDecoder(args.cell_type, EMB_DIM, DEC_HID_DIM, num_layers=N_LAYERS, dropout=DEC_DROPOUT)

    generator = Generator(DEC_HID_DIM, len(TGT.vocab))
    if args.ln:
        if args.model == 'seq2seq':
            layernorm = LayerNorm(feature=ENC_HID_DIM)
        elif args.model == 'pmi_seq2seq':
            layernorm = LayerNorm(feature=DEC_HID_DIM, conditional=True, condition_size=len(TGT.vocab),
                                condition_hidden_size=DEC_HID_DIM, condition_activation="ReLU")
        else:
            raise ValueError(args.model, "is not a legal model name!")
    else:
        layernorm = None

    if args.model == 'seq2seq':
        model = RNNBaseSeq2Seq(enc, dec, src_embedding, tgt_embedding, generator).to(device)
        train_pmi = None
    elif args.model == 'pmi_seq2seq':
        # ??????pmi_hid_dim = ENC_HID_DIM, ??????dec_hid_dim?????????enc_hid_dim????????????
        model = RNNBasePMISeq2Seq(ENC_HID_DIM, enc, dec, src_embedding, tgt_embedding, generator, layernorm).to(device)
        from scipy import sparse
        train_pmi = sparse.load_npz(os.path.join(args.dataset_dir_path, "train_sparse_pmi_matrix.npz"))
        #  valid_pmi = sparse.load_npz(os.path.join(args.dataset_dir_path, "valid_sparse_pmi_matrix.npz")) # ???????????????valid???test pmi??????????????????????????????
        #  test_pmi = sparse.load_npz(os.path.join(args.dataset_dir_path, "test_sparse_pmi_matrix.npz"))

    if args.model_path is not None:
        logger.info(f"Restore model from {args.model_path}...")
        #  model.load_state_dict(torch.load(args.model_path, map_location={'cuda:0': 'cuda:' + str(args.gpu)}))
        model = torch.load(args.model_path, map_location={'cuda:0': 'cuda:' + str(args.gpu)})
        model.to(args.device)

    print(model)
    weight = torch.ones(len(TGT.vocab), device=args.device)
    weight[TGT_PAD_IDX] = 0
    criterion = nn.NLLLoss(reduction='sum', ignore_index=TGT_PAD_IDX, weight=weight)
    #  criterion = LabelSmoothing(args, len(TGT.vocab), padding_idx=TGT_PAD_IDX, smoothing=args.smoothing)


    if args.inference:
        try:
            assert args.model_path is not None
        except AssertionError:
            logger.error("If you want to do inference, you must offer a trained model's path!")
        finally:
            inference(args, model, dataset['valid_iterator'], fields=dataset['fields'], mode='valid', pmi=train_pmi)
            inference(args, model, dataset['test_iterator'], fields=dataset['fields'], mode='test', pmi=train_pmi)
            return 0

    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = AdamOptimizer(model.parameters(), lr=args.lr, weight_decay=args.l2, max_grad_norm=args.clip)
    #  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    if args.warmup > 0:
        optimizer = NoamOptimWrapper(args.hid_dim, 1, args.warmup, optimizer)
    if args.global_step > 0:
        logger.info(f'Global step start from {args.global_step}')
        optimizer._step = args.global_step

    # TODO ??????hard-code?????????????????????
    best_global_step = 0
    best_valid_loss = float('inf')
    best_test_loss = float('inf')
    global_step = optimizer._step
    patience = args.patience if args.patience else float('inf')
    no_improve = 0

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_metrics = train(args, model, dataset['train_iterator'], optimizer, criterion, fields=dataset['fields'], writer=writer, pmi=train_pmi)
        global_step += len(dataset['train_iterator'])
        args.global_step = global_step
        valid_metrics = evaluate(args, model, dataset['valid_iterator'], criterion, fields=dataset['fields'], pmi=train_pmi)
        test_metrics = evaluate(args, model, dataset['test_iterator'], criterion, fields=dataset['fields'], pmi=train_pmi)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Global step: {global_step} | Time: {epoch_mins}m {epoch_secs}s')
        for metrics, mode in zip([train_metrics, valid_metrics, test_metrics], ['Train', 'Valid', 'Test']):
            print_metrics(metrics, mode=mode)

        # TODO ????????????logfile,??????hard-code??????
        if args.save:
            write_metrics_to_writer(valid_metrics, writer, global_step, mode='Valid')
            write_metrics_to_writer(test_metrics, writer, global_step, mode='Test')
            best_valid_loss = valid_metrics['epoch_loss']  if valid_metrics['epoch_loss'] < best_valid_loss else best_valid_loss
            best_test_loss = test_metrics['epoch_loss'] if test_metrics['epoch_loss'] < best_test_loss else best_test_loss
            best_global_step = global_step if valid_metrics['epoch_loss'] == best_valid_loss else best_global_step

            if best_global_step == global_step:
                torch.save(model, os.path.join(save_dir, f'model_global_step-{global_step}.pt'))
                #  torch.save(model.state_dict(), os.path.join(save_dir, f'model_global_step-{global_step}.pt'))
                no_improve = 0
            else:
                no_improve += 1
            with open(os.path.join(save_dir, f'log_global_step-{global_step}.txt'), 'w') as log_file:
                valid_metrics['Best Global Step'] = best_global_step
                valid_metrics['Best Loss'] = best_valid_loss

                test_metrics['Best Loss'] = best_test_loss
                test_metrics['Best PPL'] = math.exp(best_test_loss)

                inference(args, model, dataset['valid_iterator'], fields=dataset['fields'], mode='valid', pmi=train_pmi)
                inference(args, model, dataset['test_iterator'], fields=dataset['fields'], mode='test', pmi=train_pmi)

                valid_path_hyp = os.path.join(args.save_dir, 'responses-valid.txt')
                test_path_hyp = os.path.join(args.save_dir, 'responses-test.txt')
                valid_path_ref = os.path.join(args.save_dir, 'answers-valid.txt')
                test_path_ref = os.path.join(args.save_dir, 'answers-test.txt')

                other_valid_metrics = calc_metrics(path_refs=valid_path_ref, path_hyp=valid_path_hyp)
                other_test_metrics = calc_metrics(path_refs=test_path_ref, path_hyp=test_path_hyp)
                valid_metrics.update(other_valid_metrics)
                test_metrics.update(other_test_metrics)

                os.remove(os.path.join(args.save_dir, 'posts-valid.txt'))
                os.remove(os.path.join(args.save_dir, 'posts-test.txt'))
                os.remove(valid_path_hyp)
                os.remove(valid_path_ref)
                os.remove(test_path_hyp)
                os.remove(test_path_ref)

                for metric, performance in valid_metrics.items():
                    log_file.write(f'Valid {metric}: {performance}\n')
                for metric, performance in test_metrics.items():
                    log_file.write(f'Test {metric}: {performance}\n')
            if no_improve >= patience:
                break


if __name__ == '__main__':
    main()
