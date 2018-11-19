from keras.models import Sequential, load_model

import numpy as np

from keras.preprocessing import sequence
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


model = load_model('my_model.h5')

text = "<unk>\ntrade\nparty\nisin"

words = text.split('\n')

X_test = []

for word in words:
    sample = []
    for i, chara in enumerate(word):
        sample.append(np.array([ord(chara)]))
    X_test.append(np.array(sample))

X_test = np.array(X_test)
print(X_test)
max_review_length = 10
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


Xtest_padded = sequence.pad_sequences(X_test, maxlen=max_review_length)
print('result')
print(np.argmax(model.predict(Xtest_padded), axis=1))


