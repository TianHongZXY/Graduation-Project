import numpy as np
import dill
import spacy
import math
from tqdm import tqdm
from scipy import sparse
import jieba

spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def jieba_tokenize(text):
    return [token for token in jieba.cut(text)]

def calc_pmi(file_path, src_vocab_path, tgt_vocab_path):
    with open(src_vocab_path, 'rb') as f:
        src_vocab = dill.load(f) 
    with open(tgt_vocab_path, 'rb') as f:
        tgt_vocab = dill.load(f) 
    px = np.zeros(shape=(len(src_vocab), 1))
    py = np.zeros(shape=(len(tgt_vocab), 1))
    p_xy = np.zeros(shape=(len(src_vocab), len(tgt_vocab)))
    pmi_matrix = np.zeros(shape=(len(src_vocab), len(tgt_vocab)))
    corpus_size = 0

    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines()):
            dialog = line.strip().split('\t')
            try:
                assert len(dialog) == 2
            except AssertionError:
                print(line)
                print(dialog)
                continue
            src, tgt = dialog[0], dialog[1]
            corpus_size += 1
            #  src_words = tokenize_en(src)
            #  tgt_words = tokenize_en(tgt)
            src_words = jieba_tokenize(src)
            tgt_words = jieba_tokenize(tgt)
            for x in src_words:
                idx_of_x = src_vocab.stoi[x]
                px[idx_of_x] += 1
                for y in tgt_words:
                    idx_of_y = tgt_vocab.stoi[y]
                    py[idx_of_y] += 1
                    p_xy[idx_of_x][idx_of_y] += 1
    px /= corpus_size
    py /= corpus_size
    p_xy /= corpus_size
    px_sparse = sparse.csr_matrix(px)
    py_sparse = sparse.csr_matrix(py.T)
    print('px_sparse shape: ', px_sparse.shape)
    print('py_sparse shape: ', py_sparse.shape)
    px_dot_py = px_sparse.dot(py_sparse)
    #  print("px_dot_py", px_dot_py)
    #  print('px_dot_py shape: ', px_dot_py.shape)
    p_xy += 1e-12
    px_dot_py = px_dot_py.todense() + 1e-12
    print("p_xy", p_xy)
    print("px_dot_py", px_dot_py)
    pmi_matrix = p_xy / px_dot_py
    print('pmi before', pmi_matrix)
    pmi_matrix = np.maximum(np.log2(pmi_matrix), 0)
    print('pmi', pmi_matrix)
    print('pmi_matrix shape: ', pmi_matrix.shape)
    #  for i in tqdm(range(len(src_vocab))):
    #      for j in range(len(tgt_vocab)):
    #          pmi_matrix[i][j] = max(np.log2((p_xy[i][j] + 1e-12) / (px[i] * py[j] + 1e-12)), 0)

    sparse.save_npz("sparse_pmi_matrix", sparse.csr_matrix(pmi_matrix))
    #  np.save("pmi_matrix", pmi_matrix)


if __name__ == '__main__':
    #  calc_pmi(file_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/src_tgt_train.tsv",
    #           src_vocab_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/src_vocab",
    #           tgt_vocab_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/tgt_vocab")

    calc_pmi(file_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/src_tgt_train.tsv",
             src_vocab_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/src_vocab", 
             tgt_vocab_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/tgt_vocab") 


