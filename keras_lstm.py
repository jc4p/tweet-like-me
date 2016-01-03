from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras import backend
from collections import Counter
import numpy as np
import sys
from os.path import isfile

batch_size = 128
vocab_size = 256
net_size = 512
dropout = 0.30

f = open('tweets.txt')
all_lines = f.readlines()
f.close()

chars = set(" ".join(all_lines))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

max_len = np.max([len(s) for s in all_lines])

def get_most_common_first_words():
    first_words = [x.split(" ")[0].lower() for x in all_lines if " " in x]
    return [x[0] for x in Counter(first_words).most_common(50)]

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

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

X = np.zeros((len(all_lines), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(all_lines), max_len, len(chars)), dtype=np.bool)
for i, sentence in enumerate(all_lines):
    for t, char in enumerate(sentence[:-1]):
        X[i, t, char_indices[char]] = 1
    for t, char in enumerate(sentence[1:]):
        y[i, t, char_indices[char]] = 1

backend = backend._BACKEND

if isfile('keras_' + backend + '_model.json'):
    model = model_from_json(open('keras_' + backend + '_model.json').read())
else:
    model = Sequential()
    model.add(Embedding(len(chars), net_size, input_length=max_len))
    model.add(LSTM(net_size, return_sequences=True))
    #model.add(LSTM(net_size, return_sequences=True, input_shape=(max_len, len(chars))))
    model.add(Dropout(dropout))
    model.add(LSTM(net_size, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.9))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # f = open('keras_' + backend + '_model.json', 'w')
    # json_string = model.to_json()
    # f.write(json_string)
    # f.close()

first_words = get_most_common_first_words()
for iteration in range(1, 60):
    print "Iteration {}".format(iteration)
    model.fit(X, y, batch_size=batch_size, nb_epoch=1)
    model.save_weights('keras_' + backend + '_weights.h5', overwrite=True)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print 'Temp: {}'.format(diversity)

        sentence = np.random.choice(first_words) + " "
        print "Seeding with " + sentence

        for i in range(100):
            x = np.zeros((1, max_len, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence += next_char
        print sentence