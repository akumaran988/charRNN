import numpy as np
import re
import itertools
import numpy as np
from keras.layers import Dense, Flatten, LSTM
import re
from keras.models import Sequential, load_model
from keras.preprocessing import sequence as sq
from keras.layers import Dropout
from keras.utils import to_categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



#generate dict
local_dict = {}
lines = open('Text/wordList.txt', 'r').readlines()
for i, line in enumerate(lines):
    line = re.sub(r'\n', '', line)
    for word in line.split():
        local_dict[word] = i

print(local_dict)


lines = open('Text/wordsToBeFound.txt', 'r').readlines()
sequences = []
sequences_with_key_pair = []
for line in lines:
    temp = []
    sequence_with_key_pair = []
    word_key_pair = line.split('=')
    for word in word_key_pair[0].split():
        temp.append(local_dict[word])
    sequence_with_key_pair.append(temp)
    sequences.append(np.array(temp))
    word_key_pair[1] = word_key_pair[1].strip()
    for target_value in word_key_pair[1]:
        target_value = re.sub('\n', '', target_value)
    sequence_with_key_pair.append(target_value)
    sequences_with_key_pair.append(sequence_with_key_pair)
sequences = np.array(sequences)
print(sequences)
print('$$$$')
print(sequences_with_key_pair)
print('$$$$')
X_train = []
X_for_y_train = []
for sequence in sequences:
    for L in range(0, len(sequence)+1):
        for subset in itertools.combinations(sequence, L):
            if len(subset) > 0:
                elements = []
                X_for_y_elements = []
                for element in subset:
                    elements.append(np.array([element]))
                    X_for_y_elements.append(np.array(element))
                X_train.append(np.array(elements))
                X_for_y_train.append(np.array(X_for_y_elements))
X_train = np.array(X_train)
X_for_y_train = np.array(X_for_y_train)
y_train = np.zeros(X_train.shape[0])
i = 0
for row in X_for_y_train:
    row = np.array(row)
    for seq in sequences:
        t_row = []
        for r in row:
            t_row.append(r)
        t_seq = []
        for s in seq:
            t_seq.append(s)    
        
        t_row_np = np.array(t_row)
        t_seq_np = np.array(t_seq)
        
        diff = np.array_equal(t_row_np, t_seq_np)
        value = 0
        if (len(t_row) == len(t_seq) and diff):
            # value extraction
            for key_pair in sequences_with_key_pair:
                if (np.array_equal(np.array(key_pair[0]), t_seq_np)):
                    value = int(key_pair[1])
            y_train[i] = value
    i = i+1

max_review_length = 4
X_train = sq.pad_sequences(X_train, maxlen=max_review_length) 
X_test = X_train
print(X_train)
print(y_train)
y_train = to_categorical(y_train, 4)
y_test = y_train
    


# model

model = Sequential()
model.add(LSTM(600, input_shape=(4,1), return_sequences=True))
model.add(LSTM(400, return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1500, batch_size=20)

# Final evaluation of the model 
scores = model.evaluate(X_test, y_test, verbose=0) 

print("Accuracy: %.2f%%" % (scores[1]*100))

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


model.save('my_sequence_finder.h5')




