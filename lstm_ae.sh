#!/bin/bash

dir='data_categ'

./seq2seq.py -u 100 -t gru -l 3 -e 50 \
	     --word_dropout 0.2 \
	     --denoising_rate 0.2 \
	     --direction uni \
	     --attention False \
	     --max-source-sentence 100 \
	     --validation-source $dir/test_title.txt \
	     --validation-target $dir/test_title.txt \
	     $dir/train_title.txt $dir/train_title.txt \
	     $dir/train_title.vocab $dir/train_title.vocab
