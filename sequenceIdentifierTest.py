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

max_review_length = 4
X_test = np.array([np.array([[1], [4]])])
X_test = np.array(X_test)
print(X_test)
X_test = sq.pad_sequences(X_test, maxlen=max_review_length)


Xtest_padded = sq.pad_sequences(X_test, maxlen=max_review_length)
print('result')

from keras.models import load_model
model = load_model('my_sequence_finder_all.h5')

print(np.argmax(model.predict(Xtest_padded), axis=1))



