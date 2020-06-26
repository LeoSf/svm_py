from __future__ import print_function, division
from builtins import range

from sklearn.svm import SVC
from util import getKaggleMNIST
from datetime import datetime

# get the data: https://www.kaggle.com/c/digit-recognizer
Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()

#model = SVC()
model = SVC(C=5., gamma=.05)

t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print("train duration:", datetime.now() - t0)

t0 = datetime.now()
print("train score:", model.score(Xtrain, Ytrain), "duration:", datetime.now() - t0)

t0 = datetime.now()
print("test score:", model.score(Xtest, Ytest), "duration:", datetime.now() - t0)


# Outputs: 
# model = SVC()
# (svm) D:\Repos\courses\svm>python svm_mnist.py
# train duration: 0:04:18.895413
# train score: 0.9891219512195122 duration: 0:07:47.970313
# test score: 0.985 duration: 0:00:11.319317

# model = SVC(C=5., gamma=.05)
# (svm) D:\Repos\courses\svm>python 2.1_svm_mnist.py
# train duration: 0:14:40.831508
# train score: 1.0 duration: 0:12:51.490385
# test score: 0.974 duration: 0:00:18.497260