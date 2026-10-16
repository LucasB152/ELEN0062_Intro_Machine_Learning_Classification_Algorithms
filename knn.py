"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from data import make_dataset
from plot import plot_boundary


# Put your functions here
# ...
def generate_set():
    X, y = make_dataset(1500)
    # Learning set
    X_ls, y_ls = X[:1200], y[:1200]
    # Training set
    X_ts, y_ts = X[1200:], y[1200:]
    return X_ls, y_ls, X_ts, y_ts

if __name__ == "__main__":
    # 1, 9, 45, 225, 901
    pass  # Make your experiments here
    X_ls, y_ls, X_ts, y_ts = generate_set()
    for k in [1, 9, 45, 225, 901]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_ls, y_ls)
        plot_boundary(fname=f'knn_{k}', fitted_estimator=model, X=X_ts, y=y_ts, title='graph_knn_{k}')