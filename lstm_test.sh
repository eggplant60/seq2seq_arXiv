#!/bin/bash

dir='data_lstm_ae'

./seq2seq.py -u 100 -t gru -l 3 -e 1 \
	     --attention True \
	     --direction uni \
	     --word_dropout 0.0 \
	     --denoising_rate 0.0 \
	     --log-interval 1 \
	     --validation-interval 100 \
	     --max-source-sentence 300 \
	     --validation-source $dir/test_title.txt.mini \
	     --validation-target $dir/test_title.txt.mini \
	     $dir/test_title.txt.mini $dir/test_title.txt.mini \
	     $dir/train_title.vocab $dir/train_title.vocab
