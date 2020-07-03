""" 
How to implement an SVM with projected gradient descent
LDM

"""

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from util import get_spiral, get_xor, get_donut, get_clouds, plot_decision_boundary

import numpy as np
import matplotlib.pyplot as plt


# kernels
def linear(X1, X2, c=0):
    """ Dot product

    :param X1: vector 1
    :type X1: list[float]
    :param X2: vector 2
    :type X2: list[float]
    :param c: optinal C parameter, defaults to 0
    :type c: int, optional
    :return: dot product value
    :rtype: float
    """
    return X1.dot(X2.T) + c


def rbf(X1, X2, gamma=None):
    """ RBF kernel 

    :param X1: input X1   
    :type X1: one or multi -dimensional array
    :param X2: input X2
    :type X2: one or multi -dimensional array
    :param gamma: [description], defaults to None
    :type gamma: [type], optional
    :return: [description]
    :rtype: [type]
    """
    if gamma is None:
        gamma = 1.0 / X1.shape[-1]  # 1 / D
    # gamma = 0.05
    # gamma = 5. # for donut and spiral
    if np.ndim(X1) == 1 and np.ndim(X2) == 1:
        result = np.exp(-gamma * np.linalg.norm(X1 - X2)**2)
    elif (np.ndim(X1) > 1 and np.ndim(X2) == 1) or (np.ndim(X1) == 1 and np.ndim(X2) > 1):
        result = np.exp(-gamma * np.linalg.norm(X1 - X2, axis=1)**2)
    elif np.ndim(X1) > 1 and np.ndim(X2) > 1:
        result = np.exp(-gamma * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2)**2)
    return result


def sigmoid(X1, X2, gamma=0.05, c=1):
    """ Sigmoid kernel

    :param X1: input X1   
    :type X1: one or multi -dimensional array
    :param X2: input X2
    :type X2: one or multi -dimensional array
    :param gamma: gamma parameter as a function of the std deviation, defaults to 0.05
    :type gamma: float, optional
    :param c: [description], defaults to 1
    :type c: int, optional
    :return: [description]
    :rtype: [type]
    """
    return np.tanh(gamma * X1.dot(X2.T) + c)


class SVM:
    """ SVM class
    """
    def __init__(self, kernel, C=1.0):
        """ SVM class constructor

        :param kernel: kernel function
        :type kernel: function 
        :param C: miss-classification penalty, defaults to 1.0
        :type C: float, optional
        """
        self.kernel = kernel
        self.C = C

    def _train_objective(self):
        """ Function for calculating the objective. Only over the training set.
            Note: maximazing
            -- to minimize (loss funtion style) reverse the signs
        :return: [description]
        :rtype: [type]
        """
        return np.sum(self.alphas) - 0.5 * np.sum(self.YYK * np.outer(self.alphas, self.alphas))

    def fit(self, X, Y, lr=1e-5, n_iters=400):
        """ Fitting the model

        :param X: train  inputs
        :type X: list
        :param Y: training outputs
        :type Y: list
        :param lr: learning rate, defaults to 1e-5
        :type lr: float, optional
        :param n_iters: number of iteration to execute the fitting process, defaults to 400
        :type n_iters: int, optional
        """
        # we need these to make future predictions
        self.Xtrain = X
        self.Ytrain = Y
        self.N = X.shape[0]
        self.alphas = np.random.random(self.N)
        self.b = 0

        # kernel matrix
        self.K = self.kernel(X, X)
        self.YY = np.outer(Y, Y)
        self.YYK = self.K * self.YY

        # gradient ascent
        losses = []
        for _ in range(n_iters):
            loss = self._train_objective()
            losses.append(loss)
            grad = np.ones(self.N) - self.YYK.dot(self.alphas)
            self.alphas += lr * grad

            # clip
            self.alphas[self.alphas < 0] = 0
            self.alphas[self.alphas > self.C] = self.C

        # distrbution of bs
        idx = np.where((self.alphas) > 0 & (self.alphas < self.C))[0]
        bs = Y[idx] - (self.alphas * Y).dot(self.kernel(X, X[idx]))     # biases
        self.b = np.mean(bs)

        plt.plot(losses)
        plt.title("loss per iteration")
        plt.show()

    def _decision_function(self, X):
        return (self.alphas * self.Ytrain).dot(self.kernel(self.Xtrain, X)) + self.b

    def predict(self, X):
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)


def medical():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, rbf, 1e-3, 200


def medical_sigmoid():
    data = load_breast_cancer()
    X, Y = data.data, data.target
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, sigmoid, 1e-3, 200


def xor():
    X, Y = get_xor()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
    return Xtrain, Xtest, Ytrain, Ytest, kernel, 1e-2, 300

def donut():
    X, Y = get_donut()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
    return Xtrain, Xtest, Ytrain, Ytest, kernel, 1e-2, 300

def spiral():
    X, Y = get_spiral()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    kernel = lambda X1, X2: rbf(X1, X2, gamma=5.)
    return Xtrain, Xtest, Ytrain, Ytest, kernel, 1e-2, 300

def clouds():
    X, Y = get_clouds()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, linear, 1e-5, 400


if __name__ == '__main__':
    # Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = medical()
    # Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = medical_sigmoid()
    # Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = xor()
    # Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = donut()
    Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = spiral()
    # Xtrain, Xtest, Ytrain, Ytest, kernel, lr, n_iters = clouds()
    print("Possible labels:", set(Ytrain))

    # make sure the targets are (-1, +1)
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1

    # scale the data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # now we'll use our custom implementation
    model = SVM(kernel=kernel, C=1.0)

    t0 = datetime.now()
    model.fit(Xtrain, Ytrain, lr=lr, n_iters=n_iters)
    print("train duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("train score:", model.score(Xtrain, Ytrain),
          "duration:", datetime.now() - t0)
    t0 = datetime.now()
    print("test score:", model.score(Xtest, Ytest),
          "duration:", datetime.now() - t0)

    if Xtrain.shape[1] == 2:
        plot_decision_boundary(model)
