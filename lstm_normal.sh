#!/bin/bash

dir='data_seq2seq'

./seq2seq.py -u 200 -t gru -l 3 -e 50 \
	     --word_dropout 0.0 \
	     --denoising_rate 0.0 \
	     --direction uni \
	     --attention True \
	     --max-source-sentence 200 \
	     --validation-source $dir/test_abst.txt.mini \
	     --validation-target $dir/test_title.txt.mini \
	     $dir/train_abst.txt $dir/train_title.txt \
	     $dir/train_abst.vocab $dir/train_title.vocab
