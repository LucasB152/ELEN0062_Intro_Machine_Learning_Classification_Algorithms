"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from matplotlib import pyplot as plt

from data import make_dataset
from plot import plot_boundary


if __name__ == "__main__":

    depths = [1, 3, 6, 12, None]

    for depth in depths:
        train_acc, test_acc = [], []
        for seed in range(43,48):
            inputs, outputs = make_dataset(1500, random_state=seed)
            inputs_train, outputs_train = inputs[:1200], outputs[:1200]
            inputs_test, outputs_test = inputs[1200:], outputs[1200:]
            

            decision_tree_classifier = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=depth, random_state=0)
            decision_tree_classifier.fit(inputs_train, outputs_train)

            train_acc.append(accuracy_score(outputs_train, decision_tree_classifier.predict(inputs_train)))
            test_acc.append(accuracy_score(outputs_test, decision_tree_classifier.predict(inputs_test)))
            print(f"Depth={depth}: train={train_acc[-1]:.3f}, test={test_acc[-1]:.3f}")

        mean = np.mean(test_acc)
        standard_deviation = np.std(test_acc)
        current_depth = decision_tree_classifier.get_depth()
        print(f"Depth={current_depth}: mean test accuracy={mean}, standard deviation test accuracy={standard_deviation}")

        file_name = f"decision_tree_results/tree_depth_{depth}"
        plot_boundary(file_name, decision_tree_classifier, inputs_test, outputs_test, title=f"max_depth={depth}")