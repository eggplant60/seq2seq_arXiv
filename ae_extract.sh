#!/bin/bash

dir='data_lstm_ae'

./extract_vector.py -u 100 -t gru -l 3 \
	     -r 'result_ae_0122/model.npz' \
	     --word_dropout 0.0 \
	     --denoising_rate 0.2 \
	     --direction uni \
	     --attention False \
	     --max-source-sentence 100 \
	     --validation-source $dir/test_title.txt.mini \
	     --validation-target $dir/test_title.txt.mini \
	     $dir/train_title.vocab $dir/train_title.vocab
