from glob import glob 
import pandas as pd
import numpy as np
import os
import string
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import random
import sys
import io


with open('CaptionsClean.txt') as f:
    data = f.readlines()

data  = [x.replace('\n', '').split(' - ', 1) for x in data]
df = pd.DataFrame(data)
df.columns = ['meme', 'text']
text = '|'.join(list(df.text.values))
chars = sorted(list(set(text)))

print('corpus length:', len(text))
print('total chars:', len(chars))

# map indices to unique characters and vice versa
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

'''
# extract partially overlapping sequences of length maxlen,
# one-hot encode them,
# and pack them in a 3D Numpy array x of shape (sequences, maxlen, unique_characters)
'''
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
stateful = False
dropout = 0.2

model = Sequential()
model.add(LSTM(512, input_shape=(maxlen, len(chars)),
               stateful=stateful, dropout=dropout,
               return_sequences=True))
model.add(LSTM(512, stateful=stateful, dropout=dropout, return_sequences=True))
model.add(LSTM(512, stateful=stateful, dropout=dropout, return_sequences=True))
model.add(LSTM(512, stateful=stateful, dropout=dropout))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print(model.summary())


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    # select text at a random seed, generate text based on temperature randomness
    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.7, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for _ in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

filepath=os.path.join('/output', "model_checkpoint.h5")

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
earlystop = EarlyStopping(patience=3, verbose=1)

model.fit(x, y, 
           validation_split=0.2,
           batch_size=256,
           epochs=60,
           verbose=2,
           shuffle=True,
           callbacks=[
                     print_callback,
                     checkpoint,
                     earlystop
                     ])

# serialize model to JSON
model_json = model.to_json()
with open(os.path.join('/output', "model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(os.path.join('/output', "model.h5"))
print("Saved model to disk")
