#!/bin/bash

dir='data_categ'

./seq2seq.py -u 50 -t gru -l 2 -e 50 \
	     --word_dropout 0.2 \
	     --denoising_rate 0.045 \
	     --direction uni \
	     --attention False \
	     --max-source-sentence 100 \
	     --validation-source $dir/test_title.txt \
	     --validation-target $dir/test_title.txt \
	     $dir/train_title.txt $dir/train_title.txt \
	     $dir/train_title.vocab $dir/train_title.vocab

	     #--loss_type sigmoid \
