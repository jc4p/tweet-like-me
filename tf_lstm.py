import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

from collections import Counter
import numpy as np
import time
import os

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

def get_batches(batch_size):
    corpus = all_lines[:]
    np.random.shuffle(corpus)
    batches = []
    has_left = True
    while has_left:
        batch = np.zeros((batch_size, max_len), dtype=np.int)
        added_to_batch = 0
        while added_to_batch < batch_size:
            if len(corpus) == 0:
                has_left = False
                break
            d = corpus.pop(0)
            if len(d) > 10:
                for i in range(len(d)):
                    batch[added_to_batch][i] = char_indices[d[i]]
                if len(d) < max_len:
                    for i in range(len(d), max_len):
                        batch[added_to_batch][i] = 0
                added_to_batch += 1
        batches.append(batch)
    return batches


# Model gutted from https://github.com/sherjilozair/char-rnn-tensorflow
class CharRNN():
    def __init__(self, vocab_size=256, rnn_size=512, num_layers=2, batch_size=64, seq_length=max_len, learning_rate=0.001, grad_clip=10, dropout_rate=0.30):
        base_cell = rnn_cell.BasicLSTMCell(rnn_size)

        self.cell = cell = rnn_cell.MultiRNNCell([base_cell] * num_layers)

        self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
                inputs = tf.split(1, seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(x, [1]) for x in inputs]

        outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
        after_dropout = tf.nn.dropout(output, dropout_rate)

        self.logits = tf.nn.xw_plus_b(after_dropout, softmax_w, softmax_b)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * seq_length])],
                vocab_size)
        self.cost = tf.reduce_sum(loss) / batch_size / seq_length
        self.final_state = states[-1]
        self.lr = tf.Variable(learning_rate, trainable=False)
        tvars = tf.trainable_variables()
        grads = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)[0]
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def sample(self, sess, chars, vocab, count=100, prime='The ', batch_size=32):
        state = self.cell.zero_state(batch_size, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros((batch_size, max_len))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        sentence = prime
        char = prime[-1]
        for n in range(count):
            x = np.zeros((batch_size, max_len))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]
            sample = weighted_pick(p)
            pred = indices_char[sample]
            sentence += pred
            char = pred
        return sentence

def train(num_epochs, batch_size):
    learning_rate = 0.001
    learning_rate_decay = 0.95
    first_words = get_most_common_first_words()

    with tf.Session() as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state('models')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)

        for e in range(num_epochs):
            session.run(tf.assign(net.lr, learning_rate * (learning_rate_decay ** e)))
            batches = get_batches(batch_size)
            state = np.array(net.initial_state.eval())
            for batch_i in range(len(batches)):
                start = time.time()
                x = np.array(batches[batch_i])
                y = np.copy(x)
                y[:-1] = x[1:]
                y[-1] = x[0]
                x = x.reshape(batch_size, -1)
                y = y.reshape(batch_size, -1)
                feed = {net.input_data: x, net.targets: y, net.initial_state: state}
                train_loss, state, _ = session.run([net.cost, net.final_state, net.train_op], feed)
                end = time.time()
                if batch_i % 50 == 0:
                    print "epoch {}: {}/{}, train_loss = {:.3f}, time/batch = {:.3f}" \
                            .format(e, batch_i, len(batches), train_loss, end - start)
            if e % 10 == 0:
                for i in range(3):
                    print net.sample(session, indices_char, char_indices, 120, np.random.choice(first_words) + " ", batch_size)
                    print ""
                checkpoint_path = os.path.join('models', 'tf.ckpt')
                saver.save(session, checkpoint_path, global_step=e)
                print "model saved to {}".format(checkpoint_path)

def print_samples(length, count):
    first_words = get_most_common_first_words()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state('models')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            for i in range(count):
                prime = np.random.choice(first_words) + " "
                print net.sample(sess, indices_char, char_indices, length, prime)
                print ""


if __name__ == "__main__":
    batch_size = 32
    net = CharRNN(vocab_size=len(chars), rnn_size=len(chars), batch_size=batch_size)

    train(100, batch_size)
    # print_samples(120, 5)