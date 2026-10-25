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
from sklearn.tree import DecisionTreeClassifier

from data import make_dataset, make_xor_dataset
from plot import plot_boundary
from nb import NaiveBayesClassifier

SEEDS = [1, 9, 45, 225, 901]
PLOT_SEED = 42
N_FOLDS = 5

DEPTH_VALUES = [1, 3, 6, 12, None]
K_VALUES = [1, 9, 45, 225, 901]
H_VALUES = [0.005, 0.03, 0.2, 0.8, 3.2]

METHODS = {
    "Decision tree": (
        DEPTH_VALUES,
        lambda d: DecisionTreeClassifier(criterion="entropy", splitter="best",
                                         max_depth=d, random_state=0)),
    "k-NN": (K_VALUES, lambda k: KNeighborsClassifier(n_neighbors=k)),
    "KDE Naive Bayes": (H_VALUES, lambda h: NaiveBayesClassifier(bandwidth=h)),
}


DATASETS = {
    "two_moons": make_dataset,
    "xor": make_xor_dataset,
}


if __name__ == "__main__":
    for data_name, make_data in DATASETS.items():
        print(f"\n=== {data_name} ===")

        # PART 1: tuning with K-fold cross-validation
        X, y = make_data(1500, random_state=PLOT_SEED)
        X_learning, y_learning = X[:1200], y[:1200]
        X_test, y_test = X[1200:], y[1200:]

        rng = np.random.RandomState(PLOT_SEED)
        folds = np.array_split(rng.permutation(len(y_learning)), N_FOLDS)

        best_values = {}
        for name, (candidates, make_model) in METHODS.items():

            # For each hyper-parameter, we do a 5-fold cross validation
            cv_scores = []
            for c in candidates:
                fold_accuracy = []
                for i in range(N_FOLDS):
                    # Fold i is the validation set, the other folds are the training set
                    validation_index = folds[i]
                    training_index = np.concatenate([folds[j] for j in range(N_FOLDS) if j != i])

                    model = make_model(c)
                    model.fit(X_learning[training_index], y_learning[training_index])
                    fold_accuracy.append(np.mean(model.predict(X_learning[validation_index]) == y_learning[validation_index]))

                cv_scores.append(np.mean(fold_accuracy))

            # Keep the hyper-parameter value with the highest accuracy
            best_values[name] = candidates[int(np.argmax(cv_scores))]
            print(f"{name}: CV scores = {np.round(cv_scores, 3)} -> selected {best_values[name]}")

            # We train the model on the whole learning set
            model = make_model(best_values[name])
            model.fit(X_learning, y_learning)
            plot_boundary(f"mc_{data_name}_" + name.replace(" ", "_"),
                          model, X_test, y_test, title=f"{name} (tuned: {best_values[name]})")

        # PART 2: accuracy evaluation over five generations with the choosen hyper-parameter
        print("Average test accuracy over five generations:")
        for name, (candidates, make_model) in METHODS.items():
            test_acc = []
            for seed in SEEDS:
                X, y = make_data(1500, random_state=seed)
                X_learning, y_learning = X[:1200], y[:1200]
                X_test, y_test = X[1200:], y[1200:]

                model = make_model(best_values[name])
                model.fit(X_learning, y_learning)
                test_acc.append(np.mean(model.predict(X_test) == y_test))

            print(f"{name} (selected: {best_values[name]}): "
                  f"{np.mean(test_acc):.3f} +/- {np.std(test_acc):.3f}")