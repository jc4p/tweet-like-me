# Based off of https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.objectives import categorical_crossentropy
from keras.callbacks import TensorBoard
from keras import backend as K
from collections import Counter
import numpy as np
import sys
from time import clock
from os.path import isfile

batch_size = 64
vocab_size = 256
net_size = 512
dropout = 0.20

with open('tweets.txt') as f:
    all_lines = [x.strip() for x in f.readlines()]
    np.random.shuffle(all_lines)

sess = tf.Session(config=tf.ConfigProto())#log_device_placement=True))
K.set_session(sess)

chars = set(" ".join(all_lines))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

max_len = np.max([len(s) for s in all_lines])

data_X = np.zeros((len(all_lines), max_len, len(chars)), dtype=np.bool)
data_y = np.zeros((len(all_lines), len(chars)), dtype=np.bool)

def parse_all_lines():
    for i, sentence in enumerate(all_lines):
        for t, char in enumerate(sentence[:-1]):
            data_X[i, t, char_indices[char]] = 1
        for t, char in enumerate(sentence[1:]):
            data_y[i, char_indices[char]] = 1

def get_most_common_first_words():
    first_words = [x.split(" ")[0].lower() for x in all_lines if " " in x]
    return [x[0] for x in Counter(first_words).most_common(50)]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
    # exponent_raised = tf.exp(tf.div(a, temperature))
    # matrix_X = tf.div(exponent_raised, tf.reduce_sum(exponent_raised, keep_dims=True)) 
    # matrix_U = tf.random_uniform(tf.shape(a), minval=0, maxval=1)
    # return tf.argmax(tf.sub(matrix_X, matrix_U), dimension=1)

def get_sample():
    first_words = get_most_common_first_words()
    sentence = np.random.choice(first_words)
    for i in range(100):
        x = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_char = indices_char[sample(preds, 1.0)]
        sentence = sentence[1:] + next_char 
    return sentence

backend = K._BACKEND

# These are for when I want to use TensorFlow tensors directly
# X = tf.placeholder(tf.float32, shape=(len(all_lines), max_len, len(chars)))
# y = tf.placeholder(tf.float32, shape=(len(all_lines), len(chars)))

if isfile('models/keras_' + backend + '_model.json'):
    model = model_from_json(open('models/keras_' + backend + '_model.json').read())
else:
    model = Sequential()
    # model.add(Embedding(len(chars), net_size, input_length=max_len))
    first_layer = LSTM(net_size, return_sequences=True, batch_input_shape=(batch_size, max_len, len(chars)), stateful=True)
    # Uncomment to set the tensor directly
    # first_layer.set_input(X)

    model.add(first_layer)
    model.add(Dropout(dropout))
    model.add(LSTM(net_size, return_sequences=False, stateful=True))
    model.add(Dropout(dropout))
    model.add(Dense(len(chars)))
    model.add(Activation('relu'))  # fancy af

    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.9))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    f = open('models/keras_' + backend + '_model.json', 'w')
    json_string = model.to_json()
    f.write(json_string)
    f.close()

# This is for running the model directly on TensorFlow's optimizers, not looking great though.
# loss = tf.reduce_mean(categorical_crossentropy(X, model.output))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
# with sess.as_default():
#     epoch = 0
#     index = 0
#     # for i in range(10):
#     for batch in range(batch_size):
#         end_index = index + batch_size
#         if end_index > len(all_lines):
#             end_index = index + (len(all_lines) - index)
#         feed_dict = {X: data_X[index : index + batch_size], y: data_Y[index : index + batch_size]}
#         index = index + batch_size
#         if index > len(all_lines):
#             index = 0
#             print "epoch {} complete".format(epoch)
#         train_step.run(feed_dict=feed_dict)

first_words = get_most_common_first_words()
for iteration in range(1, 100):
    print "Iteration {}".format(iteration)

    # Comment the next two lines out if it shouldn't start back-up
    if isfile('models/keras_' + backend + '_weights.h5'):
        model.load_weights('models/keras_'+ backend + '_weights.h5')

    index = 0
    # TODO: Make this spit out the loss every once in a while (train_on_batch returns it)
    while True:
        end_index = index + batch_size
        if end_index > len(all_lines):
            end_index = index + (len(all_lines) - index)
        model.train_on_batch(data_X[index : index + batch_size], data_y[index : index + batch_size])
        index = index + batch_size
        if index >= len(all_lines):
            break
    print "Iteration {} done".format(iteration)

    # The above while loop does the same thing as this line, but Keras's .fit doesn't work
    # with the stateful model, there's a fit_generator but all I found about it are github
    # issues saying it doesn't work :P
    # model.fit(data_X, data_y, batch_size=batch_size, nb_epoch=1, shuffle=True,
        # callbacks=[TensorBoard(log_dir='./tf-logs', histogram_freq=0, write_graph=True)])

    model.save_weights('models/keras_' + backend + '_weights.h5', overwrite=True)

    if iteration % 20 == 0:
        for diversity in [0.5, 1.0]:
            print 'Temp: {}'.format(diversity)

            sentence = np.random.choice(first_words) + " "
            print "Seeding with " + sentence

            x = np.zeros((len(all_lines), max_len + 1, len(chars)), dtype=np.bool)
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1

            for i in range(max_len):
                preds = model.predict(x, batch_size=batch_size, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                x[0, len(sentence), char_indices[next_char]] = 1
                sentence += next_char
            print sentence

    # Clean-up and re-shuffle before next run
    model.reset_states()
    np.random.shuffle(all_lines)
    data_X = np.zeros((len(all_lines), max_len, len(chars)), dtype=np.bool)
    data_y = np.zeros((len(all_lines), len(chars)), dtype=np.bool)
    parse_all_lines()