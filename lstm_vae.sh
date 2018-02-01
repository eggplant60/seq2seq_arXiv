#!/bin/bash

dir='data_categ'

./seq2seq_vae.py -u 100 \
		 -t gru \
		 -l 1 \
		 -e 40 \
		 -b 10 \
		 --n_embed 150 \
	     --word_dropout 0.38 \
	     --denoising_rate 0.05 \
	     --n_latent 12 \
	     --validation-interval 1000 \
	     --max-source-sentence 10 \
	     --validation-source $dir/test_title.txt \
	     --validation-target $dir/test_title.txt \
	     $dir/train_title.txt $dir/train_title.txt \
	     $dir/train_title.vocab $dir/train_title.vocab
#	     --validation-source $dir/train_title.txt.mini \
#	     --validation-target $dir/train_title.txt.mini \
