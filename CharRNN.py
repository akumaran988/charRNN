import numpy as np
from keras.layers import Dense, Flatten, LSTM
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.utils import to_categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#%%


#%%
def train_seq_generator(filename, value=None):
    common_words  = open(filename).readlines()

    Common_X_train = []

    for common_word in common_words:
        sample = []
        for i, chara in enumerate(common_word):
            sample.append(np.array([ord(chara)]))
        Common_X_train.append(np.array(sample))

    Common_X_train = np.array(Common_X_train)
    #print(Common_X_train)

    if(value != None):
        Common_y_train = np.zeros((Common_X_train.shape[0]))
        for i in range(Common_y_train.shape[0]):
            Common_y_train[i] = value
        Common_y_train = to_categorical(Common_y_train, 5)
    
        return Common_X_train, Common_y_train
    else:
        Common_y_train = np.zeros((Common_X_train.shape[0]))
        Common_y_train = to_categorical(Common_y_train, 5)
    
        return Common_X_train, Common_y_train

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

pX_train, py_train = train_seq_generator('Text/words.txt')

X_train, y_train = train_seq_generator('Text/trade.txt', 1)

print(pX_train.shape)
print(X_train.shape)
print(py_train.shape)
print(y_train.shape)

X_train = np.concatenate((pX_train, X_train), axis=0)
y_train = np.concatenate((py_train, y_train), axis=0)

pX_train, py_train = shuffle_in_unison(X_train, y_train)

print(pX_train.shape)
print(X_train.shape)
print(py_train.shape)
print(y_train.shape)

X_train, y_train = train_seq_generator('Text/party.txt', 2)
X_train = np.concatenate((pX_train, X_train), axis=0)
y_train = np.concatenate((py_train, y_train), axis=0)

pX_train, py_train = shuffle_in_unison(X_train, y_train)

print(pX_train.shape)
print(X_train.shape)
print(py_train.shape)
print(y_train.shape)

X_train, y_train = train_seq_generator('Text/currency.txt', 4)
X_train = np.concatenate((pX_train, X_train), axis=0)
y_train = np.concatenate((py_train, y_train), axis=0)

pX_train, py_train = shuffle_in_unison(X_train, y_train)

print(pX_train.shape)
print(X_train.shape)
print(py_train.shape)
print(y_train.shape)
X_test = X_train
y_test = y_train
max_review_length = 10 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


print(X_train.shape)
print(y_train.shape)

 

# create the model 
model = Sequential()
model.add(LSTM(500, input_shape=(10,1), return_sequences=True))
model.add(LSTM(500, return_sequences=True))
model.add(LSTM(500))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=300)

# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))

text = "trade date\ncurrency\nfor\nqwert\nprice"

words = text.split('\n')

X_test = []

for word in words:
    sample = []
    for i, chara in enumerate(word):
        sample.append(np.array([ord(chara)]))
    X_test.append(np.array(sample))

X_test = np.array(X_test)
print(X_test)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


Xtest_padded = sequence.pad_sequences(X_test, maxlen=max_review_length)
print('result')
print(model.predict(Xtest_padded))

from keras.models import load_model

model.save('my_model.h5')

#del model  # deletes the existing model
#%%
# returns a compiled model
# identical to the previous one
