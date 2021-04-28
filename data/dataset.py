from torchtext.data import BucketIterator
from torchtext import data, datasets
from torchtext.vocab import Vectors
import logging
import os
import re
from collections import Counter, OrderedDict
import logging
import coloredlogs
import jieba
import spacy

spacy_en = spacy.load('en_core_web_sm')
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def whitespace_tokenize(text):
    return [tok for tok in text.split()]


def re_whitespace_tokenize(text):
    text = re.sub('[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', " ", text)
    return [tok for tok in text.split()]


def jieba_tokenize(text):
    return [token for token in jieba.cut(text)]


def chinese_char_tokenize(text):
    return list(text)


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


def load_iwslt(args):
    # For data loading.
    import spacy
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    BATCH_SIZE = args.bs
    MIN_FREQ = args.min_freq
    MAX_LEN = args.maxlen

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"
    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD,
                     # batch_first=True,
                     lower=True,
                     include_lengths=True,
                     )
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD,
                     # batch_first=True,
                     lower=True,
                     include_lengths=True,
                     )

    train, valid, test = datasets.Multi30k.splits(
        exts=('.de', '.en'), fields=[('src', SRC), ('tgt', TGT)],
        # filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
        #                       len(vars(x)['tgt']) <= MAX_LEN,
    )

    SRC.build_vocab(train, min_freq=MIN_FREQ)
    TGT.build_vocab(train, min_freq=MIN_FREQ)
    logger.info(f"SRC has {len(SRC.vocab.itos)} words.")
    logger.info(f"TGT has {len(TGT.vocab.itos)} words.")
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        shuffle=True,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True,
        device=args.device)

    return {"fields": {'src': SRC, 'tgt': TGT}, "vocab": SRC.vocab, "train_data": train, "val_data": valid,
            "test_data": test, "train_iterator": train_iterator,
            "valid_iterator": valid_iterator, "test_iterator": test_iterator}


import dill
def dump_examples(args, train, valid, test):
    # 保存examples
    with open(os.path.join(args.dataset_dir_path, 'train_examples'), 'wb') as f:
        dill.dump(train.examples, f)
    with open(os.path.join(args.dataset_dir_path, 'valid_examples'), 'wb') as f:
        dill.dump(valid.examples, f)
    with open(os.path.join(args.dataset_dir_path, 'test_examples'), 'wb') as f:
        dill.dump(test.examples, f)


def load_examples(args):
    # 加载examples
    with open(os.path.join(args.dataset_dir_path, 'train_examples'), 'rb') as f:
        train_examples = dill.load(f)
    with open(os.path.join(args.dataset_dir_path, 'valid_examples'), 'rb') as f:
        valid_examples = dill.load(f)
    with open(os.path.join(args.dataset_dir_path, 'test_examples'), 'rb') as f:
        test_examples = dill.load(f)
    return train_examples, valid_examples, test_examples


def seq2seq_dataset(args, is_train=True, tokenizer=None):
    MIN_FREQ = args.min_freq
    BATCH_SIZE = args.bs
    MAX_LEN = args.maxlen
    pad_token = '<pad>'
    sos_token = '<sos>'
    eos_token = '<eos>'
    # TODO 抽象tokenizer为一个类，具有属性类名和tokenize函数,解决下面的硬编码提示使用的是什么tokenizer
    if tokenizer is None:
        logger.info("Tokenizer is not given! Using spacy tokenizer as default.")
        tokenizer = tokenize_en
    else:
        logger.info(f"Using given tokenizer {args.tokenizer}")
    SRC = data.Field(tokenize=tokenizer, pad_token=pad_token,
                     include_lengths=True, 
                     #  batch_first=True,
                     lower=True,
                     )
    TGT = data.Field(tokenize=tokenizer, init_token=sos_token,
                     eos_token=eos_token, pad_token=pad_token,
                     include_lengths=True, 
                     #  batch_first=True,
                     lower=True,
                     )
    # fields = [('src', SRC), ('tgt', TGT), ('cue', SRC)]
    fields = [('src', SRC), ('tgt', TGT)]

    if MAX_LEN:
        filter_pred = lambda x: len(vars(x)['src']) > 0 and len(vars(x)['tgt']) > 0 and len(vars(x)['src']) <= MAX_LEN and len(vars(x)['tgt']) <= MAX_LEN
    else:
        filter_pred = lambda x: len(vars(x)['src']) > 0 and len(vars(x)['tgt']) > 0

    if args.use_serialized:
        train_examples, valid_examples, test_examples = load_examples(args)
        train = data.Dataset(examples=train_examples, fields=fields, filter_pred=filter_pred)
        valid = data.Dataset(examples=valid_examples, fields=fields, filter_pred=filter_pred)
        test = data.Dataset(examples=test_examples, fields=fields, filter_pred=filter_pred)
    else:
        dataset = data.TabularDataset.splits(path=args.dataset_dir_path, format='TSV', train=args.train_file,
                                             validation=args.valid_file, test=args.test_file, fields=fields,
                                             filter_pred=filter_pred,
                                             )
        train, valid, test = dataset
        if args.serialize:
            dump_examples(args, train, valid, test)
    logger.info(f"Total {len(train)} pairs in train data")
    logger.info(f"Total {len(valid)} pairs in valid data")
    logger.info(f"Total {len(test)} pairs in test data")
    vectors = None
    if args.pretrained_embed_file is not None:
        logger.info(f"Using pretrained vectors {args.pretrained_embed_file}")
        if not os.path.exists('.vector_cache'):
            os.mkdir('.vector_cache')
            vectors = Vectors(name=args.pretrained_embed_file)
    if not args.use_serialized:
        logger.info("No pre-defined vocab given, building new vocab now...")
        # train disc的时候保存vocab文件，到了train gen时加载，保证两者的embedding共用一个vocab
        SRC.build_vocab(train,
                      max_size=args.max_vocab_size,
                      min_freq=MIN_FREQ,
                      vectors=vectors
                      )
        TGT.build_vocab(train,
                        max_size=args.max_vocab_size,
                        min_freq=MIN_FREQ,
                        vectors=vectors)
        if args.serialize:
            with open(os.path.join(args.dataset_dir_path, 'src_vocab'), 'wb') as f:
                dill.dump(SRC.vocab, f)
            with open(os.path.join(args.dataset_dir_path, 'tgt_vocab'), 'wb') as f:
                dill.dump(TGT.vocab, f)

    else:
        #  优先使用给定的vocab file
        logger.info(f"Using vocab file from {args.dataset_dir_path}")
        with open(os.path.join(args.dataset_dir_path, 'src_vocab'), 'rb') as f:
            SRC.vocab = dill.load(f)
        with open(os.path.join(args.dataset_dir_path, 'tgt_vocab'), 'rb') as f:
            TGT.vocab = dill.load(f)

    logger.info(f"SRC has {len(SRC.vocab.itos)} words.")
    logger.info(f"TGT has {len(TGT.vocab.itos)} words.")
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train, valid, test),
        batch_size=BATCH_SIZE,
        shuffle=not args.inference,
        sort_key=lambda x: len(x.src),
        sort_within_batch=True,
        device=args.device)

    return {"fields": {'src': SRC, 'tgt': TGT}, "train_data": train, "val_data": valid,
            "test_data": test, "train_iterator": train_iterator,
            "valid_iterator": valid_iterator, "test_iterator": test_iterator}

