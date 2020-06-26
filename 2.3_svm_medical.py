# sklearn.svm: Support Vector Machines
# svm.SVC(*[, C, kernel, degree, gamma, …])
# C-Support Vector Classification.

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# load the data
data = load_breast_cancer()

# split the data into train and test sets
# this lets us simulate how our model will perform in the future
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data.data, data.target, test_size=0.33)

# scale the data
scaler = StandardScaler()
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

# model = SVC(kernel='linear')
model = SVC(kernel='rbf')
# model = SVC()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

# Output: [linear]
# (svm) D:\Repos\courses\svm>python 2.3_svm_medical.py
# train score: 0.9921259842519685
# test score: 0.9414893617021277

# Output: [rbf]
# (svm) D:\Repos\courses\svm>python 2.3_svm_medical.py
# train score: 0.9868766404199475
# test score: 0.9680851063829787