from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

# load the data
data = load_breast_cancer()

# trying different hyper-parameters using a pipeline
for C in (0.5, 1.0, 5.0, 10.0):
  # Pipeline
  pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(C=C))])
  # k-fold cross-validation
  # cv=int, cross-validation generator or an iterable, default=None
  scores = cross_val_score(pipeline, data.data, data.target, cv=5)
  # printing results
  print("C:", C, "mean:", scores.mean(), "std:", scores.std())

# output: 
# (svm) D:\Repos\courses\svm>python 2.5_crossval.py
# C: 0.5 mean: 0.9683744760130415 std: 0.017175387281311592
# C: 1.0 mean: 0.9736376339077782 std: 0.014678541667933545
# C: 5.0 mean: 0.9789318428815401 std: 0.006990390328940835   <-- best result
# C: 10.0 mean: 0.9771774569166279 std: 0.008921310582673642