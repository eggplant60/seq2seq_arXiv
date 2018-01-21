#!/bin/bash

dir='data_lstm_ae'

./seq2seq.py -u 100 -t gru -l 3 -e 50 \
	     --dropout_rate 0.0 \
	     --direction uni \
	     --attention False \
	     --max-source-sentence 100 \
	     --validation-source $dir/test_title.txt.mini \
	     --validation-target $dir/test_title.txt.mini \
	     $dir/train_title.txt $dir/train_title.txt \
	     $dir/train_title.vocab $dir/train_title.vocab
