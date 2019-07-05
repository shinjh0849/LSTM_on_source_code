'''
#Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import tensorflow as tf

device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print ('Found GPU at: {}'.format(device_name))

F_OUT = open('numpy_output.txt', 'w')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: ' + str(epoch))
    F_OUT.write('----- Generating text after Epoch: ' + str(epoch) + '\n')

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:' + str(diversity))
        F_OUT.write('----- diversity:' + str(diversity) + '\n')

        generated = ''
        # sentence = text[start_index: start_index + maxlen]
        sentence = 'os.path.dirname(os.path.abspath(cfuncs._'
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        F_OUT.write('----- Generating with seed: "' + sentence + '"' + '\n')
        # sys.stdout.write(generated)
        F_OUT.write(generated)

        for i in range(1000):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            F_OUT.write(next_char)
            # sys.stdout.write(next_char)
            # sys.stdout.flush()
        print()
        F_OUT.write('\n')

# with tf.device('/gpu:0'):

with open('./numpy_ascii.txt', encoding='utf-8') as f:
    text = f.read().lower()
    
print('text length:', len(text))
print('first 100:', text[:50])

#   path = get_file(
#       'nietzsche.txt',
#       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#   with io.open(path, encoding='utf-8') as f:
#       text = f.read().lower()

print(len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
print(chars)
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('sentences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
print(x.shape)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
print(y.shape)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
# model.add(CuDNNLSTM(128, input_shape=(maxlen, len(chars))))
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)




print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y,
        batch_size=128,
        epochs=20,
        callbacks=[print_callback])

F_OUT.close()

print('done!')