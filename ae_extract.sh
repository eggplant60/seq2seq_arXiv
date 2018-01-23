#!/bin/bash

dir='data_categ'

./extract_vector.py -u 100 -t gru -l 3 \
	     -r 'result_ae_0122_2/model.npz' \
	     --word_dropout 0.0 \
	     --denoising_rate 0.0 \
	     --direction uni \
	     --attention False \
	     --max-source-sentence 100 \
	     --validation-source $dir/test_title.txt \
	     --validation-target $dir/test_title.txt \
	     $dir/train_title.vocab $dir/train_title.vocab
