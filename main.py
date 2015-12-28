# From the ApolloCaffee samples:
# https://github.com/Russell91/apollocaffe/blob/769011/examples/apollocaffe/char_model/char_model.py

import numpy as np
import os
import json
import theano.tensor as T
import theano

import apollocaffe
from apollocaffe.layers import (Concat, Dropout, LstmUnit, InnerProduct, NumpyData,
    Softmax, SoftmaxWithLoss, Wordvec)

batch_size = 32
vocab_size = 256
zero_symbol = vocab_size - 1
dimension = 250
base_lr = 0.15
clip_gradients = 10
i_temperature = 1.2
dropout_rate = 0.30

parser = apollocaffe.base_parser()
args = parser.parse_args()
apollocaffe.set_device(args.gpu)
apollocaffe.set_random_seed(0)

i_temp_x = T.fmatrix('x')
i_temp_z = i_temp_x * i_temperature
i_temp_f = theano.function([i_temp_x], i_temp_z)


def get_data():
    epoch = 0
    while True:
        with open('tweets.txt', 'r') as f:
            all_lines = f.readlines()
            numpy.random.shuffle(all_lines)
            for x in all_lines:
                if len(x) > 10:  # Some arbitary limit to ignore blank and non-sensical tweets
                    yield x
        print('epoch %s finished' % epoch)
        epoch += 1

def get_data_batch():
    data_iter = get_data()
    while True:
        batch = []
        for i in range(batch_size):
            d = next(data_iter)
            while not d:
                d = next(data_iter)
            batch.append(d)
        yield batch

def pad_batch(sentence_batch):
    max_len = max(len(x) for x in sentence_batch)
    result = []
    for sentence in sentence_batch:
        chars = [min(ord(c), 140) for c in sentence] 
        result.append(chars + [zero_symbol] * (max_len - len(sentence)))
    return np.array(result)

def forward(net, sentence_batches):
    net.clear_forward()
    batch = next(sentence_batches)
    sentence_batch = pad_batch(batch)
    length = min(sentence_batch.shape[1], 100)
    assert length > 0

    net.f(NumpyData('lstm_seed', np.zeros((batch_size, dimension))))
    for step in range(length):
        if step == 0:
            prev_hidden = 'lstm_seed'
            prev_mem = 'lstm_seed'
            word = np.zeros(sentence_batch[:, 0].shape)
        else:
            prev_hidden = 'lstm%d_hidden' % (step - 1)
            prev_mem = 'lstm%d_mem' % (step - 1)
            word = sentence_batch[:, step - 1]
        net.f(NumpyData('word%d' % step, word))
        net.f(Wordvec('wordvec%d' % step, dimension, vocab_size,
            bottoms=['word%d' % step], param_names=['wordvec_param']))
        net.f(Concat('lstm_concat%d' % step, bottoms=[prev_hidden, 'wordvec%d' % step]))
        net.f(LstmUnit('lstm%d' % step,
            bottoms=['lstm_concat%d' % step, prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm%d_hidden' % step, 'lstm%d_mem' % step],
            num_cells=dimension))
        net.f(Dropout('dropout%d' % step, dropout_rate,
            bottoms=['lstm%d_hidden' % step]))

        net.f(NumpyData('label%d' % step, sentence_batch[:, step]))
        net.f(InnerProduct('ip%d' % step, vocab_size,
            bottoms=['dropout%d' % step],
            param_names=['softmax_ip_weights', 'softmax_ip_bias']))
        net.f(SoftmaxWithLoss('softmax_loss%d' % step, ignore_label=zero_symbol,
            bottoms=['ip%d' % step, 'label%d' % step]))

def softmax_choice(data):
    probs = data.flatten().astype(np.float64)
    probs /= probs.sum()
    return np.random.choice(range(len(probs)), p=probs)

def eval_forward(net):
    net.clear_forward()
    output_words = []
    net.f(NumpyData('lstm_hidden_prev', np.zeros((1, dimension))))
    net.f(NumpyData('lstm_mem_prev', np.zeros((1, dimension))))
    length = 140
    for step in range(length):
        net.clear_forward()
        net.f(NumpyData('word', [0]))
        prev_hidden = 'lstm_hidden_prev'
        prev_mem = 'lstm_mem_prev'
        if step == 0:
            output = ord(' ')
        else:
            output = softmax_choice(net.blobs['softmax'].data)
        output_words.append(output)
        net.blobs['word'].data[0] = output
        net.f(Wordvec('wordvec', dimension, vocab_size,
            bottoms=['word'], param_names=['wordvec_param']))
        net.f(Concat('lstm_concat', bottoms=[prev_hidden, 'wordvec']))
        net.f(LstmUnit('lstm', dimension,
            bottoms=['lstm_concat', prev_mem],
            param_names=['lstm_input_value', 'lstm_input_gate',
                'lstm_forget_gate', 'lstm_output_gate'],
            tops=['lstm_hidden_next', 'lstm_mem_next']))
        net.f(Dropout('dropout', dropout_rate, bottoms=['lstm_hidden_next']))

        net.f(InnerProduct('ip', vocab_size, bottoms=['dropout'],
            param_names=['softmax_ip_weights', 'softmax_ip_bias']))
        net.blobs['ip'].data[:] = i_temp_f(net.blobs['ip'].data)
        net.f(Softmax('softmax', bottoms=['ip']))
        net.blobs['lstm_hidden_prev'].data_tensor.copy_from(
            net.blobs['lstm_hidden_next'].data_tensor)
        net.blobs['lstm_mem_prev'].data_tensor.copy_from(
            net.blobs['lstm_mem_next'].data_tensor)
    print ''.join([chr(x) for x in output_words])

net = apollocaffe.ApolloNet()
net.load('tweets.caffemodel')
sentence_batches = get_data_batch()

forward(net, sentence_batches)
train_loss_hist = []

display = 50
loggers = [apollocaffe.loggers.TrainLogger(display),
    apollocaffe.loggers.SnapshotLogger(1000, '/tmp/char')]
for i in range(10000):
    forward(net, sentence_batches)
    train_loss_hist.append(net.loss)
    net.backward()
    lr = (base_lr * (0.8)**(i // 2500))
    net.update(lr, clip_gradients=clip_gradients)
    for logger in loggers:
        logger.log(i, {'train_loss': train_loss_hist,
            'apollo_net': net, 'start_iter': 0})
    if i % display == 0:
        eval_forward(net)
    if i % 200:
        net.save('tweets.caffemodel')
