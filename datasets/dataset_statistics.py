import sys
import numpy as np


def dataset_statistics(path, src_tgt_delimiter='\t', turns_delimiter='EOS', max_n=4e6, language='en'):
    # author: Xiang Gao @ Microsoft Research, Oct 2018
    print(path)
    src_lens = []
    tgt_lens = []
    sum_src_turns = 0
    n = 0
    for line in open(path, encoding='utf-8'):
        n += 1
        line = line.strip('\n')
        if src_tgt_delimiter is not None:
            src, tgt = line.split(src_tgt_delimiter)
            #  sum_src_turns += len(src.split(turns_delimiter))
            if language == 'en':
                src_lens.append(len(src.split()))
            elif language == 'zh':
                src_lens.append(len(list(src)))
            else:
                raise ValueError("language must be en or zh!")
        else:
            tgt = line
        if language == 'en':
            tgt_lens.append(len(tgt.split()))
        elif language == 'zh':
            tgt_lens.append(len(list(tgt)))
        else:
            raise ValueError("language must be en or zh!")
        if n%1e6 == 0:
            print('checked %i M'%(n/1e6))
        if n == max_n:
            break

    src_len_90 = sorted(src_lens)[int(n*0.9)]
    tgt_len_90 = sorted(tgt_lens)[int(n*0.9)]

    print('total checked = %i (%.3f M)'%(n, n/1e6))
    #  print('src_turns: avg = %.2f'%(sum_src_turns/n))
    print('src_len: avg = %.2f, 90 percent = %i'%(np.mean(src_lens), src_len_90))
    print('tgt_len: avg = %.2f, 90 percent = %i'%(np.mean(tgt_lens), tgt_len_90))


if __name__ == '__main__':
    dataset_statistics(path=sys.argv[1], language=sys.argv[2]) 
