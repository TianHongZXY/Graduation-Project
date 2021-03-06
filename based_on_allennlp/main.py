from allennlp.data import Vocabulary
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.data_loaders import SimpleDataLoader, MultiProcessDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.modules.token_embedders import Embedding, TokenCharactersEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import GruSeq2VecEncoder
from dataset_reader import Seq2SeqDatasetReader
from model import create_seq2seqmodel
from train import train
import torch
import argparse
import os
from datetime import datetime
from dateutil import tz, zoneinfo
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='conversational seq2seq'
    )

    parser.add_argument('--gpu', default=-1, type=int, help='which GPU to use, -1 means using CPU')
    parser.add_argument('--save', action="store_true", help='whether to save model or not')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--emb_dim', default=300, type=int, help='embedding dim')
    parser.add_argument('--hid_dim', default=500, type=int, help='hidden dim of lstm')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio')
    parser.add_argument('--n_epochs', default=30, type=int, help='num of train epoch')
    parser.add_argument('--clip', default=None, type=float, help='grad clip')
    parser.add_argument('--maxlen', default=29, type=int, help='fixed length of text')
    parser.add_argument('--train_file', default=None, type=str, help='train file path')
    parser.add_argument('--valid_file', default=None, type=str, help='valid file path')
    parser.add_argument('--test_file', default=None, type=str, help='test file path')
    parser.add_argument('--save_dir', default='models', type=str, help='save dir')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--l2', default=0, type=float, help='l2 regularization')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    args, unparsed = parser.parse_known_args()

    device = torch.device(args.gpu if (torch.cuda.is_available() and args.gpu >= 0) else 'cpu')

    source_token_indexers = {"tokens": SingleIdTokenIndexer(namespace="source_tokens"),
                             #  "character_tokens": TokenCharactersIndexer(namespace="source_char_tokens"),
                             }
    target_token_indexers = {"tokens": SingleIdTokenIndexer(namespace="target_tokens"),
                             #  "character_tokens": TokenCharactersIndexer(namespace="target_char_tokens"),
                             }
    
    dataset_reader = Seq2SeqDatasetReader(source_token_indexers=source_token_indexers,
                                          target_token_indexers=target_token_indexers)
    train_data = list(dataset_reader.read(args.train_file))
    vocab = Vocabulary.from_instances(train_data)

    valid_data = dataset_reader.read(args.valid_file)
    test_data = dataset_reader.read(args.test_file)
    
    src_embedding = Embedding(embedding_dim=args.emb_dim,
                          vocab_namespace="source_tokens",
                          vocab=vocab)
    #  src_char_embedding = Embedding(embedding_dim=args.emb_dim,
    #                             vocab_namespace="source_char_tokens",
    #                             vocab=vocab)
    #  src_char_encoder = TokenCharactersEncoder(embedding=src_char_embedding,
    #                                            encoder=GruSeq2VecEncoder(input_size=args.emb_dim,
    #                                                                      hidden_size=args.hid_dim))
    tgt_embedding = Embedding(embedding_dim=args.emb_dim,
                          vocab_namespace="target_tokens",
                          vocab=vocab)
    #  tgt_char_embedding = Embedding(embedding_dim=args.emb_dim,
    #                             vocab_namespace="target_char_tokens",
    #                             vocab=vocab)
    #  tgt_char_encoder = TokenCharactersEncoder(embedding=tgt_char_embedding,
    #                                            encoder=GruSeq2VecEncoder(input_size=args.emb_dim,
    #                                                                      hidden_size=args.hid_dim))
    src_embedders = BasicTextFieldEmbedder({
        "tokens": src_embedding,
        #  "character_tokens": src_char_encoder
        })
    # tgt_embedders = BasicTextFieldEmbedder({
    #     "tokens": tgt_embedding,
        #  "character_tokens": tgt_char_encoder
        # })
    
    train_loader = SimpleDataLoader.from_dataset_reader(
                                                      reader=dataset_reader, 
                                                      data_path=args.train_file,
                                                      batch_size=args.bs,
                                                      shuffle=True)
    train_loader.index_with(vocab)
    val_loader = SimpleDataLoader.from_dataset_reader(reader=dataset_reader,
                                                      data_path=args.valid_file,
                                                      batch_size=args.bs)
    val_loader.index_with(vocab)
    model = create_seq2seqmodel(vocab, src_embedders=src_embedders, tgt_embedders=tgt_embedding, hidden_dim=args.hid_dim,
                                max_decoding_steps=args.maxlen, device=device)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {count_parameters(model)} parameters.")

    save_dir = None
    if args.save:
        tz_sh = tz.gettz('Asia/Shanghai')
        save_dir = os.path.join(args.save_dir, 'run' + str(datetime.now(tz=tz_sh)).replace(":", "-").split(".")[0].replace(" ", '.'))
        args.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    train(args, model=model, dataset_reader=dataset_reader, num_epochs=args.n_epochs,
          train_loader=train_loader, val_loader=val_loader, test_data=test_data, serialization_dir=save_dir, device=device)
