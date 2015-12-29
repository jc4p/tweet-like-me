from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
import numpy as np
import sys

batch_size = 32
max_len = 140
vocab_size = 256
zero_symbol = 0

def get_data():
    epoch = 0
    while True:
        with open('tweets.txt', 'r') as f:
            all_lines = f.readlines()
            np.random.shuffle(all_lines)
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
    result = []
    for sentence in sentence_batch:
        chars = [min(ord(c), max_len) for c in sentence] 
        result.append(chars + [zero_symbol] * (max_len - len(sentence)))
    return np.array(result)


model = Sequential()
model.add(Embedding(vocab_size, vocab_size, 250, input_length=max_len, mask_zero=True))
model.add(LSTM(250, return_sequences=True))
model.add(Dropout(0.15))
model.add(LSTM(250, return_sequences=False))
model.add(Dropout(0.15))
model.add(Dense(250))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05, momentum=0.9))

# This code doesn't run right now since I don't have
# any X or y yet. I haven't figured out if I should be
# doing vectorization or using that Embedding layer, and how.
for iteration in range(1, 60):
    print 'Iteration {}'.format(iteration)
    model.fit(X, y, batch_size=32, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print '----- diversity: {}'.format(diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print '----- Generating with seed: "{}"'.format(sentence)
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()