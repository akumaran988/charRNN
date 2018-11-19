import numpy
import re
import itertools
import numpy as np
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

'''
a = numpy.asarray([[1, 1], [2, 2], [3, 3]])
b = numpy.asarray([[1], [2], [3]])
print(a)
print(b)
a, b = shuffle_in_unison(a, b)

print(a)
print(b)
import re
wordList = open('Text/wordList.txt', 'r').readlines()
numerical_word_list = []
for i in range(len(wordList)):
    print(wordList[i])
    wordList[i] = re.sub('\n','', wordList[i])
    numerical_word_list.append(i)

print(wordList)
print(numerical_word_list)

X_train = []
for L in range(0, len(numerical_word_list)+1):
    for subset in itertools.combinations(numerical_word_list, L):
        X_train.append(subset)
        print(np.array(subset))

X = np.array([np.array([2, 3, 4]),np.array([4, 5, 6])])

X_ex = []
y_ex = []
for i, row in enumerate(X):
    print('X' + str(X[i]) + str(i))
for i, row in enumerate(X):
    temp = []
    r = np.zeros((row.shape[0], 8))
    for i, col in enumerate(row):
        row_c = 0
        for j in np.arange(col-0.4, col+0.4, 0.1):
            if(row_c >= 8):
                continue
            r[i,row_c] = j
            row_c = row_c + 1
    r = r.transpose()
    temp = r
    X_ex.append(np.array(temp))
print(X_ex)
'''



def combinations(X, l):
    temp = []
    for i in range(len(X)):
        t = []
        if i == len(X)-l+1:
            break
        for j in np.arange(i,i+l):
            t.append(X[j])
        temp.append(t)
    return np.array(temp) 

x = [[0, 1, 0, 1, 0, 2, 0, 1]]
def splitz(inp):
    key = (0,)
    return [list(q) for k, q in itertools.groupby(inp, lambda x:x in key) if not k]
print(x)
sub = np.array(splitz(x[0]))
print(sub)
X = x[0]
y = []
l=3

for s in sub:
    temp = combinations(X,len(s))
    t = np.array(s)
    for i, te in enumerate(temp):
        if np.array_equal(t, te):
            y.append(i)
            if i == len(X)-l+1:
                break
            for p in np.arange(i, i+len(te)):
                X[p] = 0

print(y)


'''
for x in sub:
    pred = model.predict(x)
    if pred == 1:
        # retrive from index
    elif pred == 2
        # retrive from index
''' 
'''
import json

data = {}
data['isin'] = {'value':'isin_value',
                'trade-date':'10/12/18'}
json_data = json.dumps(data)

print(json_data)
'''
import pytesseract
from PIL import Image

text = pytesseract.image_to_string(Image.open('imig.png'))

print(text)
