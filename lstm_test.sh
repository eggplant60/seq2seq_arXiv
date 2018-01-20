#!/bin/bash

dir='data_lstm_ae'

./seq2seq.py -u 100 -t gru -l 3 -e 100 \
	     --direction uni \
	     --dropout_rate 0.0 \
	     --log-interval 1 \
	     --validation-interval 2 \
	     --max-source-sentence 300 \
	     --validation-source $dir/mini_title.txt \
	     --validation-target $dir/mini_title.txt \
	     $dir/mini_title.txt $dir/mini_title.txt \
	     $dir/train_title.vocab $dir/train_title.vocab

