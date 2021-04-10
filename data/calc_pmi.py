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

    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines()):
            dialog = line.strip().split('\t')
            try:
                assert len(dialog) == 2
            except AssertionError:
                print(line)
                print(dialog)
            src, tgt = dialog[0], dialog[1]
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
    print('px shape: ', px.shape)
    print('py shape: ', py.shape)
    p_xy = np.log2(p_xy + 1e-12)
    #  p_xy= sparse.csr_matrix(p_xy)
    px_dot_py = np.matmul(px, py.T) + 1e-12
    print('px_dot_py shape: ', px_dot_py.shape)
    pmi_matrix = np.maximum(0, p_xy / px_dot_py)
    print('pmi_matrix shape: ', pmi_matrix.shape)
    #  for i in tqdm(range(len(src_vocab))):
    #      for j in range(len(tgt_vocab)):
    #          pmi_matrix[i][j] = max(np.log2((p_xy[i][j] + 1e-12) / (px[i] * py[j] + 1e-12)), 0)

    sparse.save_npz("sparse_pmi_matrix", sparse.csr_matrix(pmi_matrix))
    #  np.save("pmi_matrix", pmi_matrix)


if __name__ == '__main__':
    calc_pmi(file_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/src_tgt_train.tsv", 
             src_vocab_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/src_vocab", 
             tgt_vocab_path="/home/zxy21/code_and_data/Graduation-Project/datasets/LCCC-base-split/tgt_vocab") 


