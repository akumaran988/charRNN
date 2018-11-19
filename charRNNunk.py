import numpy as np
from keras.layers import Dense, Flatten, LSTM
import re
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
    Common_y_train = []
    for j, common_word in enumerate(common_words):
        common_word = re.sub('\n','',common_word)
        sample = []
        for i, chara in enumerate(common_word):
            sample.append(np.array([ord(chara)]))
        Common_X_train.append(np.array(sample))
        Common_y_train.append(j)
    Common_X_train = np.array(Common_X_train)
    Common_y_train = to_categorical(Common_y_train, 10)
    return Common_X_train, Common_y_train
    #print(Common_X_train)

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

X_train, y_train = train_seq_generator('Text/wordList.txt')
X_test = X_train
y_test = y_train

max_review_length = 10 
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length) 
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


print(X_train.shape)
print(y_train.shape)

 

# create the model 
model = Sequential()
model.add(LSTM(500, input_shape=(10,1), return_sequences=False))
model.add(Dense(10, activation='softmax'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=3)

# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))

text = "trade\ncurrency\nfor\nqwert\nprice"

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
