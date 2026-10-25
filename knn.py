"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
from random import seed

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from data import make_dataset
from plot import plot_boundary

SEEDS = [1, 9, 45, 225, 901]
PLOT_SEED = 42
K_VALUES = [1, 9, 45, 225, 901]

def generate_set(random_state=None):
    X, y = make_dataset(1500, random_state=random_state)
    # Learning set & Test set
    return X[:1200], y[:1200], X[1200:], y[1200:]

def decision_boundary():
    X_ls, y_ls, X_ts, y_ts = generate_set(PLOT_SEED)
    for k in K_VALUES:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_ls, y_ls)
        file_name = f"knn_{k}"
        plot_boundary(file_name, fitted_estimator=model, X=X_ts, y=y_ts, title=f'graph_knn_{k}')

def eval_accuracies():
    print("# Neighbors: \tMean accuracy, \tStandard Deviation")
    for k in K_VALUES:
        acc = []
        model = KNeighborsClassifier(n_neighbors=k)
        for s in SEEDS:
            X_ls, y_ls, X_ts, y_ts = generate_set(s)
            model.fit(X_ls, y_ls)
            acc.append(model.score(X_ts, y_ts))
        print(f"K={k}: \t{np.mean(acc)} \t{np.std(acc)} ")


if __name__ == "__main__":
    decision_boundary()
    eval_accuracies()
