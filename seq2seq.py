#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json

#from nltk.translate import bleu_score
from rouge import Rouge
import numpy
import progressbar
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

from chainer import serializers
from numpy import random

UNK = 0
EOS = 1


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
                 type_unit, dropout_rate, direc, attr):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_vocab, n_units)
            self.embed_y = L.EmbedID(n_target_vocab, n_units)
            #self.attention = Attention(n_units)
            if type_unit == 'lstm':
                if direc == 'uni':
                    self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
                    self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
                elif direc == 'bi':
                    self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
                    self.decoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
            elif type_unit == 'gru':
                if direc == 'uni':
                    self.encoder = L.NStepGRU(n_layers, n_units, n_units, 0.1)
                    self.decoder = L.NStepGRU(n_layers, n_units, n_units, 0.1)
                elif direc == 'bi':
                    self.encoder = L.NStepBiGRU(n_layers, n_units, n_units, 0.1)
                    self.decoder = L.NStepBiGRU(n_layers, n_units, n_units, 0.1)
            if direc == 'uni':
                self.W = L.Linear(n_units, n_target_vocab)
            elif direc == 'bi':
                self.W = L.Linear(2*n_units, n_target_vocab)
            if attr:
                self.Wc = L.Linear(2*n_units, n_units)

        self.n_layers = n_layers
        self.n_units = n_units
        self.type_unit = type_unit
        self.dropout_rate = dropout_rate
        self.attr = attr
        

    def __call__(self, xs, ys):
        xs = [x[::-1] for x in xs]

        eos = self.xp.array([EOS], 'i')
        
        ys_drop = [self.denoiseInput(y) for y in ys]
        ys_in = [F.concat([eos, y], axis=0) for y in ys_drop]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        # Both xs and ys_in are lists of arrays.
        exs = sequence_embed(self.embed_x, xs)
        eys = sequence_embed(self.embed_y, ys_in)

        batch = len(xs)
        # None represents a zero vector in an encoder.
        if self.type_unit == 'lstm':
            hx, cx, at = self.encoder(None, None, exs)
            _, _, os = self.decoder(hx, cx, eys)
        elif self.type_unit == 'gru':
            hx, at = self.encoder(None, exs)
            _, os = self.decoder(hx, eys)

        # os: batch_size x len_of_sentence x hiddensize
        # at is same
        # print(len(hx))
        # print(hx[0].shape)
        # print(len(at))
        # print(at[0].shape)
        
        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        concat_at = F.concat(at, axis=0)
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)

        # Attention
        if self.attr:
            concat_os_new = self._calculate_attention_layer_output(concat_os, concat_at)
        else:
            concat_os_new = concat_os

        # Calcurate Loss
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os_new), concat_ys_out, reduce='no')) / batch

        chainer.report({'loss': loss.data}, self)
        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss


    def _calculate_attention_layer_output(self, c_os, c_at):
        inner_prod = F.matmul(c_os, c_at, transb=True) # 第2引数を転地
        weights = F.softmax(inner_prod)
        contexts = F.matmul(weights, c_at)
        concatenated = F.concat((contexts, c_os))
        new_embedded_output = F.tanh(self.Wc(concatenated))
        return new_embedded_output
    
    
    def denoiseInput(self, y):  ###WordDropOut
        if self.dropout_rate > 0.0:
            unk = self.xp.array([UNK], 'i')

            for i in range(len(y)):
                if random.rand() < self.dropout_rate:
                    y[i] = unk
        return y

    
    def translate(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            if self.type_unit == 'lstm':
                h, c, a = self.encoder(None, None, exs)
            elif self.type_unit == 'gru':
                h, a = self.encoder(None, exs)
            ys = self.xp.full(batch, EOS, 'i')
            result = []
            for i in range(max_length):
                eys = self.embed_y(ys)
                eys = F.split_axis(eys, batch, 0)
                if self.type_unit == 'lstm':
                    h, c, ys = self.decoder(h, c, eys)                
                elif self.type_unit == 'gru':
                    h, ys = self.decoder(h, eys)
                ca = F.concat(a, axis=0)
                cys = F.concat(ys, axis=0)
                
                # Attention
                if self.attr:
                    cys_new = self._calculate_attention_layer_output(cys, ca)
                else:
                    cys_new = cys
                wy = self.W(cys_new)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)

        # Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
        # support NumPy 1.9.
        result = cuda.to_cpu(
            self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = numpy.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

    
    def out_vector(self, xs):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed_x, xs)
            if self.type_unit == 'lstm':
                vectors, _, _ = self.encoder(None, None, exs)
            elif self.type_unit == 'gru':
                vectors, _ = self.encoder(None, exs)
                
        h = F.concat(vectors, axis=1)
        return h.data


def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=100, device=-1, max_length=100):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length
        self.rouge = Rouge()

    def __call__(self, trainer):
        with chainer.no_backprop_mode():
            references = []
            hypotheses = []
            for i in range(0, len(self.test_data), self.batch):
                sources, targets = zip(*self.test_data[i:i + self.batch])
                #references.extend([[t.tolist()] for t in targets])
                references.extend([' '.join(map(str, t.tolist())) for t in targets])

                sources = [
                    chainer.dataset.to_device(self.device, x) for x in sources]
                # ys = [y.tolist()
                #       for y in self.model.translate(sources, self.max_length)]
                ys = [' '.join(map(str, y.tolist()))
                      for y in self.model.translate(sources, self.max_length)]
                hypotheses.extend(ys)

        # bleu = bleu_score.corpus_bleu(
        #     references, hypotheses,
        #     smoothing_function=bleu_score.SmoothingFunction().method1)
        # chainer.report({self.key: bleu})
        scores = self.rouge.get_scores(hypotheses, references, avg=True)
        rouge_l = scores["rouge-l"]
        chainer.report({self.key[0]: rouge_l["p"]})
        chainer.report({self.key[1]: rouge_l["r"]})
        chainer.report({self.key[2]: rouge_l["f"]})


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = 0
    word_ids['<EOS>'] = 1
    return word_ids


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK) for w in words], 'i')
            data.append(array)
    return data


def calculate_unknown_ratio(data):
    unknown = sum((s == UNK).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('SOURCE_VOCAB', help='source vocabulary file')
    parser.add_argument('TARGET_VOCAB', help='target vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='number of sweeps over the dataset to train')
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
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=1000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--dropout_rate', '-d', type=float, default=0.3)
    parser.add_argument('--direction', choices={'uni', 'bi'}, type=str, default='uni') # bi: alpha version
    parser.add_argument('--attention', '-a', type=bool, default=False)
    args = parser.parse_args()


    source_ids = load_vocabulary(args.SOURCE_VOCAB)
    target_ids = load_vocabulary(args.TARGET_VOCAB)
    train_source = load_data(source_ids, args.SOURCE)
    train_target = load_data(target_ids, args.TARGET)
    assert len(train_source) == len(train_target)
    train_data = [(s, t)
                  for s, t in six.moves.zip(train_source, train_target)
                  if args.min_source_sentence <= len(s)
                  <= args.max_source_sentence and
                  args.min_source_sentence <= len(t)
                  <= args.max_source_sentence]
    train_source_unknown = calculate_unknown_ratio(
        [s for s, _ in train_data])
    train_target_unknown = calculate_unknown_ratio(
        [t for _, t in train_data])

    print('Source vocabulary size: %d' % len(source_ids))
    print('Target vocabulary size: %d' % len(target_ids))
    print('Train data size: %d' % len(train_data))
    print('Train source unknown ratio: %.2f%%' % (train_source_unknown * 100))
    print('Train target unknown ratio: %.2f%%' % (train_target_unknown * 100))

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    model = Seq2seq(args.layer, len(source_ids), len(target_ids), \
                    args.unit, args.type_unit, args.dropout_rate, args.direction, args.attention)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'main/perp',
         'bleu', 'rouge', 'f', 'elapsed_time']),
        trigger=(args.log_interval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}.npz'),
        trigger=(5, 'epoch'))

    if args.validation_source and args.validation_target:
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

        @chainer.training.make_extension()
        def translate(trainer):
            source, target = test_data[numpy.random.choice(len(test_data))]
            result = model.translate([model.xp.array(source)])[0]

            source_sentence = ' '.join([source_words[x] for x in source])
            target_sentence = ' '.join([target_words[y] for y in target])
            result_sentence = ' '.join([target_words[y] for y in result])
            print('#  source : ' + source_sentence)
            print('#  result : ' + result_sentence)
            print('#  expect : ' + target_sentence)

        trainer.extend(
            translate, trigger=(args.validation_interval, 'iteration'))
        trainer.extend(
            CalculateBleu(
                model, test_data, ['bleu', 'rouge', 'f'], device=args.gpu),
            trigger=(args.validation_interval, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, model)

    # # save vector
    # sources = test_source[0]
    # vector = model.out_vector([model.xp.array(sources)])
    # print(vector.shape)
    # print(vector)
    # return None
    
    print('start training')
    trainer.run()

    with open('result/args.txt', 'w') as f:
        asi = ['{}: {}'.format(i, getattr(args, i)) for i in dir(args) if not '_' in i[0]]
        f.write('\n'.join(asi))
    serializers.save_npz('result/model.npz', model)

    
if __name__ == '__main__':
    main()
