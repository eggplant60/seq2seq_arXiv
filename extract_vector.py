#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle

from chainer.cuda import to_cpu

from seq2seq import *


class Seq2seq_ride(Seq2seq):

    def out_vector(self, xs):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            if self.type_unit == 'lstm':
                vectors, _, _ = self.encoder(None, None, exs)
            elif self.type_unit == 'gru':
                vectors, _ = self.encoder(None, exs)
                
        c_vectors = F.concat(vectors, axis=1) # layer
        l_vectors = to_cpu(c_vectors.data)    # to cpu
        return l_vectors


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    # parser.add_argument('SOURCE', help='source sentence list')
    # parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    # parser.add_argument('--batchsize', '-b', type=int, default=50,
    #                     help='number of sentence pairs in each mini-batch')
    # parser.add_argument('--epoch', '-e', type=int, default=100,
    #                     help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='number of units')
    parser.add_argument('--type_unit', '-t', choices={'lstm', 'gru'},
                        help='number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--min-source-sentence', type=int, default=2, # for caluculation of ngram 2
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=500,
                        help='maximum length of source sentence')
    parser.add_argument('--min-target-sentence', type=int, default=2, # for caluculation of ngram 2
                        help='minimium length of target sentence')
    parser.add_argument('--max-target-sentence', type=int, default=50,
                        help='maximum length of target sentence')
    # parser.add_argument('--log-interval', type=int, default=200,
    #                     help='number of iteration to show log')
    # parser.add_argument('--validation-interval', type=int, default=1000,
    #                     help='number of iteration to evlauate the model '
    #                     'with validation dataset')
    parser.add_argument('--word_dropout', '-w', type=float, default=0.0)
    parser.add_argument('--denoising_rate', '-d', type=float, default=0.0)
    parser.add_argument('--direction', choices={'uni', 'bi'}, type=str, default='uni') # bi: doesn't work...
    parser.add_argument('--attention', '-a', type=bool, default=False)
    args = parser.parse_args()


    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)

    model = Seq2seq_ride(args.layer, len(source_ids), len(target_ids), args.unit,
                         args.type_unit, args.word_dropout,
                         args.denoising_rate, args.direction, args.attention)
    
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)
        
    if args.resume:
        serializers.load_npz(args.resume, model)

    test_source = load_data(source_ids, args.validation_source)
    test_target = load_data(target_ids, args.validation_target)
    assert len(test_source) == len(test_target)
    test_data = list(six.moves.zip(test_source, test_target))
    test_data = [(s, t) for s, t in test_data if 0 < len(s) and 0 < len(t)]
    test_source_unknown = calculate_unknown_ratio(
        [s for s, _ in test_data])
    test_target_unknown = calculate_unknown_ratio(
        [t for _, t in test_data])

    print('Validation data: %d' % len(test_data))
    print('Validation source unknown ratio: %.2f%%' %
          (test_source_unknown * 100))
    print('Validation target unknown ratio: %.2f%%' %
          (test_target_unknown * 100))

    # save vector    
    xs = [ model.xp.array(source) for source in test_source[0:1000]]
    vector = model.out_vector(xs)
    print(vector)

    pkl_file = 'vector.pkl'
    
    with open(pkl_file, 'wb') as f:
        pickle.dump(vector, f)

    print('dump on \'{}\''.format(pkl_file))
    
    
if __name__ == '__main__':
    main()
