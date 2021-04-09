#!/bin/bash
nohup python -u -W ignore main.py --tokenizer spacy_en --lr 0.0002 --clip 5 --emb_dim 256 --hid_dim 512 --n_epochs 200 --save --cell_type LSTM --teaching_rate 0.25 --dropout 0.5 --gpu 9 --bs 32 --dataset_dir_path datasets/dailydialog_src_tgt/ --train_file dstc_cleaned_train.tsv --valid_file dstc_cleaned_valid.tsv --test_file dstc_cleaned_test.tsv --comment compare_model1_dailydialog_ --use_serialized > dailydialog_1.log 2>&1 &
sleep 5s
nohup python -u -W ignore main.py --tokenizer spacy_en --lr 0.0002 --clip 5 --emb_dim 256 --hid_dim 512 --n_epochs 200 --save --cell_type LSTM --teaching_rate 0.5 --dropout 0.5 --gpu 8 --bs 32 --dataset_dir_path datasets/dailydialog_src_tgt/ --train_file dstc_cleaned_train.tsv --valid_file dstc_cleaned_valid.tsv --test_file dstc_cleaned_test.tsv --comment compare_model2_dailydialog_ --use_serialized > dailydialog_2.log 2>&1 &
sleep 5s
nohup python -u -W ignore main.py --tokenizer spacy_en --lr 0.0002 --clip 5 --emb_dim 256 --hid_dim 512 --n_epochs 200 --save --cell_type LSTM --teaching_rate 0.75 --dropout 0.5 --gpu 6 --bs 32 --dataset_dir_path datasets/dailydialog_src_tgt/ --train_file dstc_cleaned_train.tsv --valid_file dstc_cleaned_valid.tsv --test_file dstc_cleaned_test.tsv --comment compare_model3_dailydialog_ --use_serialized > dailydialog_3.log 2>&1 &
# nohup python -u driver/Train.py  --thread 1 --gpu 7 --config msr/default.cfg > msr.log 2>&1 &
# nohup python -u driver/Train.py  --thread 1 --gpu 4 --config quora/default.cfg  > quora.log 2>&1 &
# nohup python -u driver/Train.py  --thread 1 --gpu 6 --config sci/default.cfg   > sci.log 2>&1 &
# nohup python -u driver/Train.py  --thread 1 --gpu 7 --config sick/default.cfg  > sick.log 2>&1 &
# nohup python -u driver/Train.py  --thread 1 --gpu 5 --config snli/default.cfg  > snli.log 2>&1 &
