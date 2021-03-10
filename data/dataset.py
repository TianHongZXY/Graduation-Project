from torchtext.data import BucketIterator
from torchtext import data
from torchtext.vocab import Vectors
import logging
import os
import re
from collections import Counter, OrderedDict
import logging
import coloredlogs
import spacy
spacy_en = spacy.load('en')
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def whitespace_tokenize(text):
    return [tok for tok in text.split()]


def re_whitespace_tokenize(text):
    text = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', " ", text)
    return [tok for tok in text.split()]


def build_field_vocab(vocab_file, field, min_freq=1, vectors=None):
    d = dict()
    with open(vocab_file, 'r') as f:
        for line in f.readlines():
            line = line.split(':')
            word, freq = line[0], line[1].strip('\n')
            d[word] = int(freq)
    freqs = Counter(d)
    specials = list(OrderedDict.fromkeys(
        tok for tok in [field.unk_token, field.pad_token, field.init_token,
                        field.eos_token]))
    field.vocab = field.vocab_cls(counter=freqs,
                                  # max_size=35000,
                                  min_freq=min_freq,
                                  specials=specials,
                                  vectors=vectors)
    return field


def seq2seq_dataset(args, is_train=True, tokenizer=None):
    MIN_FREQ = args.min_freq
    BATCH_SIZE = args.bs
    MAX_LEN = args.maxlen
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    if tokenizer is None:
        print("Tokenizer is not given! Using spacy tokenizer as default.")
        tokenizer = tokenize_en
    SRC = data.Field(tokenize=tokenizer, pad_token=pad_token,
                     include_lengths=True, batch_first=True)
    TGT = data.Field(tokenize=tokenizer, init_token=sos_token,
                     eos_token=eos_token, pad_token=pad_token,
                     include_lengths=True, batch_first=True)
    # fields = [('src', SRC), ('tgt', TGT), ('cue', SRC)]
    fields = [('src', SRC), ('tgt', TGT)]
    filter_pred = None
    if MAX_LEN:
        filter_pred = lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['tgt']) <= MAX_LEN
    dataset = data.TabularDataset.splits(path=args.dataset_dir_path, format='TSV', train=args.train_file,
                                         validation=args.valid_file, test=args.test_file, fields=fields,
                                         filter_pred=filter_pred,
                                         )
    train, valid, test = dataset
    vectors = None
    if args.pretrained_embed_file is not None:
        logger.info(f"Using pretrained vectors {args.pretrained_embed_file}")
        if not os.path.exists('.vector_cache'):
            os.mkdir('.vector_cache')
            vectors = Vectors(name=args.pretrained_embed_file)
    if args.vocab_file is None:
        logger.info("No pre-defined vocab given, building new vocab now...")
        # train disc的时候保存vocab文件，到了train gen时加载，保证两者的embedding共用一个vocab
        SRC.build_vocab(train,
                      # max_size=35000,
                      min_freq=MIN_FREQ,
                      vectors=vectors
                      )
        TGT.build_vocab(train,
                        # max_size=35000,
                        min_freq=MIN_FREQ,
                        vectors=vectors)
        logger.info(f"SRC has {len(SRC.vocab.itos)} words.")
        logger.info(f"TGT has {len(TGT.vocab.itos)} words.")

        if is_train and args.save:
            # 保存时保存所有出现的单词，不管是否超过min_freq的要求
            logger.info(f"Saving vocab file to {args.save_dir}")
            with open(os.path.join(args.save_dir, 'src_vocab.txt'), 'w') as f:
                for k, v in SRC.vocab.freqs.most_common():
                    f.write("{}:{}\n".format(k, v))
            with open(os.path.join(args.save_dir, 'tgt_vocab.txt'), 'w') as f:
                for k, v in TGT.vocab.freqs.most_common():
                    f.write("{}:{}\n".format(k, v))
    else:
        #  优先使用给定的vocab file
        logger.info(f"Using vocab file from {args.save_dir}")
        SRC = build_field_vocab(vocab_file=os.path.join(args.save_dir, 'src_vocab.txt'),
                                field=SRC, min_freq=MIN_FREQ, vectors=vectors)
        TGT = build_field_vocab(vocab_file=os.path.join(args.save_dir, 'tgt_vocab.txt'),
                                field=TGT, min_freq=MIN_FREQ, vectors=vectors)
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        shuffle=True,
        sort_key=lambda x: len(x.src) + len(x.tgt),
        sort_within_batch=True,
        device=args.device)

    return {"fields": {'src': SRC, 'tgt': TGT}, "vocab": SRC.vocab, "train_data": train, "val_data": valid,
            "test_data": test, "train_iterator": train_iterator,
            "valid_iterator": valid_iterator, "test_iterator": test_iterator}
