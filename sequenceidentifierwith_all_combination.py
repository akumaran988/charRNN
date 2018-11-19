import numpy as np
import re
import itertools
import numpy as np
from keras.layers import Dense, Flatten, LSTM, GRU
import re
from keras.models import Sequential, load_model
from keras.preprocessing import sequence as sq
from keras.layers import Dropout
from keras.utils import to_categorical
from itertools import permutations 
from keras import optimizers
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



#generate dict
local_dict = {}
lines = open('Text/wordList.txt', 'r').readlines()
for i, line in enumerate(lines):
    line = re.sub(r'\n', '', line)
    for word in line.split():
        local_dict[word] = float(i)

print(local_dict)


# generating sequences of input combination to train the model

wordList = open('Text/wordList.txt', 'r').readlines()
numerical_word_list = []
for i in range(len(wordList)):
    print(wordList[i])
    wordList[i] = re.sub('\n','', wordList[i])
    numerical_word_list.append(i)

print(wordList)
print(numerical_word_list)

# Restricting the sequence length to 4
max_sequence_size = 3
X_train = []
for L in range(0, len(numerical_word_list)+1):
    for subset in itertools.combinations(numerical_word_list, L):
        if len(subset) > max_sequence_size:
            break
        permutes = permutations(subset) 
        for perm in permutes:
            if len(subset) > 0:
                X_train.append(np.array(perm))
                #print(np.array(perm))

X_train = np.array(X_train)
print(X_train.shape)

# generating y_train output vector
y_train = np.zeros(X_train.shape[0])

# Based on the file wordsToBeFound.txt find and replace output for the sequence in X_train
lines = open('Text/wordsToBeFound.txt', 'r').readlines()
sequences_with_key_pair = []
for line in lines:
    temp = []
    sequence_with_key_pair = []
    word_key_pair = line.split('=')
    for word in word_key_pair[0].split():
        temp.append(local_dict[word])
    sequence_with_key_pair.append(temp)
    word_key_pair[1] = word_key_pair[1].strip()
    for target_value in word_key_pair[1]:
        target_value = re.sub('\n', '', target_value)
    sequence_with_key_pair.append(target_value)
    sequences_with_key_pair.append(sequence_with_key_pair)

print('$$$$')
print(sequences_with_key_pair)
print('$$$$')


for i, Xinput in enumerate(X_train):
    for sequence_with_key_pair in sequences_with_key_pair:
        isequal = np.array_equal(Xinput, np.array(sequence_with_key_pair[0]))
        if isequal == True:
            y_train[i] = float(sequence_with_key_pair[1])

new_X_train = []
new_y_train = []
for i in range(X_train.shape[0]):
    new_X_train.append(X_train[i])
    new_y_train.append(y_train[i])

for p, row in enumerate(X_train):
    print(p)
    if new_y_train[p] == 0:
        continue
    temp = []
    r = np.zeros((row.shape[0], 80))
    y = np.full((80,), new_y_train[p])
    for i, col in enumerate(row):
        row_c = 0
        for j in np.arange(col-0.4, col+0.4, 0.01):
            if(row_c >= 80):
                continue
            r[i,row_c] = j
            row_c = row_c + 1
    r = r.transpose()
    for q, row in enumerate(r):
        new_X_train.append(row)
        new_y_train.append(y[q])

new_X_train = np.array(new_X_train)
new_y_train = np.array(new_y_train)
print(new_X_train)
print(new_y_train)

print(new_X_train.shape)
print(new_y_train.shape)

X_train = new_X_train
y_train = new_y_train

print(y_train)

# Changing the input shape for the LSTM to accept
LSTM_X_train = []
for Xinput in X_train:
    temp = []
    for time_stamp in Xinput:
        temp.append(np.array([time_stamp]))
    temp = np.array(temp)
    LSTM_X_train.append(temp)

LSTM_X_train = np.array(LSTM_X_train)
X_train = LSTM_X_train
max_review_length = 4
X_train = sq.pad_sequences(X_train, maxlen=max_review_length) 
X_test = X_train
print(X_train)
print(y_train)
y_train = to_categorical(y_train, 4)
y_test = y_train
    
# model

model = Sequential()
model.add(GRU(1000, input_shape=(4,1), return_sequences=True))
model.add(GRU(1000, return_sequences=True))
model.add(GRU(1000, return_sequences=False))
model.add(Dense(4, activation='sigmoid'))

optimizer = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=50)


# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))


model.save('my_sequence_finder_all.h5')

text = "trade date"

words = text.split('\n')

X_test = []

for word in words:
    sample = []
    for i, chara in enumerate(word):
        sample.append(np.array([ord(chara)]))
    X_test.append(np.array(sample))

X_test = np.array(X_test)
print(X_test)
X_test = sq.pad_sequences(X_test, maxlen=max_review_length)


Xtest_padded = sq.pad_sequences(X_test, maxlen=max_review_length)
print('result')
print(model.predict(Xtest_padded))


model.save('my_sequence_finder_all.h5')

