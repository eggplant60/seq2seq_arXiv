#!/bin/bash

dir='data_seq2seq'

./seq2seq.py -u 100 -t gru -l 3 -e 100 \
	     --dropout_rate 0.0 \
	     --direction uni \
	     --max-source-sentence 300 \
	     --validation-source $dir/test_abst.txt \
	     --validation-target $dir/test_title.txt \
	     $dir/train_abst.txt $dir/train_title.txt \
	     $dir/train_abst.vocab $dir/train_title.vocab
