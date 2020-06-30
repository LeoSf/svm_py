from __future__ import print_function, division
from future.utils import iteritems
from builtins import range, input

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from util import get_clouds

import numpy as np
import matplotlib.pyplot as plt


class LinearSVM:
    """ Linear SVM class implementation
    """
    def __init__(self, C=1.0):
        """ Init method of the class

        :param C:  relative strength of the misclassification penalty, defaults to 1.0
        :type C: float, optional
        """
        self.C = C

    def _objective(self, margins):
        """ Minimization objective function

        :param margins: margin
        :type margins: list[floats]
        :return: loss
        :rtype: float
        """
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - margins).sum()

    def _decision_function(self, X):
        """ Linear equation for the svm

        :param X: input data
        :type X: numpy matrix
        :return: linear function value
        :rtype: float
        """
        return X.dot(self.w) + self.b

    def fit(self, X, Y, lr=1e-5, n_iters=400):
        """ Function to fit the model

        :param X: training data
        :type X: list
        :param Y: labels
        :type Y: list
        :param lr: learning rate, defaults to 1e-5
        :type lr: float, optional
        :param n_iters: number of iteration to train, defaults to 400
        :type n_iters: int, optional
        """
        N, D = X.shape
        self.N = N
        self.w = np.random.randn(D)
        self.b = 0

        # gradient descent
        losses = []
        for _ in range(n_iters):
            margins = Y * self._decision_function(X)        # calculate the functional margins
            loss = self._objective(margins)
            losses.append(loss)

            idx = np.where(margins < 1)[0]                  # datapoints wich violates the svm margin line
            grad_w = self.w - self.C * Y[idx].dot(X[idx])   # gradient for w
            self.w -= lr * grad_w
            grad_b = -self.C * Y[idx].sum()                 # gradient for b
            self.b -= lr * grad_b

        self.support_ = np.where((Y * self._decision_function(X)) <= 1)[0]  # save datapoints wich lays or violates the margin
        print("num SVs:", len(self.support_))

        print("w:", self.w)
        print("b:", self.b)

        # hist of margins
        # m = Y * self._decision_function(X)
        # plt.hist(m, bins=20)
        # plt.show()

        plt.plot(losses)
        plt.title("loss per iteration")
        plt.show()

    def predict(self, X):
        """ predict value

        :param X: input data
        :type X: numpy matrix
        :return: sign value
        :rtype: signed float
        """
        return np.sign(self._decision_function(X))

    def score(self, X, Y):
        """ classifier accuracy from input data   

        :param X: input data
        :type X: numpy array
        :param Y: labels
        :type Y: list
        :return: classifier accuarcy
        :rtype: float
        """
        P = self.predict(X)
        return np.mean(Y == P)


def plot_decision_boundary(model, X, Y, resolution=100, colors=('b', 'k', 'r')):
    """ plot datapoint with separation and margins

    :param model: model to use
    :type model: sklearn model
    :param X: input data
    :type X: numpy array
    :param Y: labels
    :type Y: list
    :param resolution: [description], defaults to 100
    :type resolution: int, optional
    :param colors: colors for data, defaults to ('b', 'k', 'r')
    :type colors: tuple, optional
    """
    np.warnings.filterwarnings('ignore')
    fig, ax = plt.subplots()

    # Generate coordinate grid of shape [resolution x resolution]
    # and evaluate the model over the entire space
    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), resolution)
    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), resolution)
    grid = [[model._decision_function(np.array([[xr, yr]]))
             for yr in y_range] for xr in x_range]
    grid = np.array(grid).reshape(len(x_range), len(y_range))

    # Plot decision contours using grid and
    # make a scatter plot of training data
    ax.contour(x_range, y_range, grid.T, (-1, 0, 1), linewidths=(1, 1, 1),
               linestyles=('--', '-', '--'), colors=colors)
    ax.scatter(X[:, 0], X[:, 1],
               c=Y, lw=0, alpha=0.3, cmap='seismic')

    # Plot support vectors (non-zero alphas)
    # as circled points (linewidth > 0)
    mask = model.support_
    ax.scatter(X[:, 0][mask], X[:, 1][mask],
               c=Y[mask], cmap='seismic')

    # debug
    ax.scatter([0], [0], c='black', marker='x')

    # debug
    # x_axis = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    # w = model.w
    # b = model.b
    # # w[0]*x + w[1]*y + b = 0
    # y_axis = -(w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, y_axis, color='purple')
    # margin_p = (1 - w[0]*x_axis - b)/w[1]
    # plt.plot(x_axis, margin_p, color='orange')
    # margin_n = -(1 + w[0]*x_axis + b)/w[1]
    # plt.plot(x_axis, margin_n, color='orange')

    plt.show()


def clouds():
    """ inteface to load the "cloud dataset" 

    :return: training and testing datasets, learning rate and iterations to train
    :rtype: lists
    """
    X, Y = get_clouds()
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200


def medical():
    """ inteface to load the standard breast cancer dataset

    :return: training and testing datasets, learning rate and iterations to train
    :rtype: lists
    """
    data = load_breast_cancer()
    X, Y = data.data, data.target
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33)
    return Xtrain, Xtest, Ytrain, Ytest, 1e-3, 200


if __name__ == '__main__':
    """ Main function
    """
    Xtrain, Xtest, Ytrain, Ytest, lr, n_iters = clouds()
    print("Possible labels:", set(Ytrain))

    # make sure the targets are (-1, +1)
    Ytrain[Ytrain == 0] = -1
    Ytest[Ytest == 0] = -1

    # scale the data
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    # now we'll use our custom implementation
    model = LinearSVM(C=1.0)

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
        plot_decision_boundary(model, Xtrain, Ytrain)
