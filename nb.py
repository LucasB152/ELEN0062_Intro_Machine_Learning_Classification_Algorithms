"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math as math

from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_dataset
from plot import plot_boundary

class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    """Naive Bayes classifier."""

    def __init__(self, bandwidth=0.2):
        self.bandwidth = bandwidth

    def fit(self, X, y):
        """Fit a naive bayes classifier model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        classes, nb_element_classes = np.unique(y, return_counts=True)
        total_nb_element = len(y)

        self.classes_ = classes
        self.priors_ = nb_element_classes/total_nb_element
        self.X_per_classes_ = []

        for c in classes:
            self.X_per_classes_.append(X[y == c])

        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        scores = self.compute_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]


    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        return np.exp(self.predict_log_proba(X))
    

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """

        scores = self.compute_scores(X)
        log_proba = np.zeros_like(scores)

        for a in range(scores.shape[0]):
            log_px = self.log_sum_exp(scores[a])   
            for k in range(scores.shape[1]):
                log_proba[a, k] = scores[a, k] - log_px

        return log_proba

    def compute_scores(self, X):
        X = np.asarray(X, dtype=float)
        h = self.bandwidth

        result = []

        for input_x in X:
            row = []
            for prior, x_values in zip(self.priors_, self.X_per_classes_):

                # Compute KDE
                nb_element = len(x_values)
                log_likelihood = 0.0

                # We loop over each feature
                for i in range(len(input_x)):
                    log_kernels = []

                    # We loop over each data in the dataset corresponding to that class
                    for x_value in x_values:
                        log_kernels.append(self.LogGaussianKernel((input_x[i] - x_value[i]) / h))

                    log_likelihood += self.log_sum_exp(log_kernels) - math.log(nb_element * h)

                row.append(math.log(prior) + log_likelihood)

            result.append(row)

        return np.array(result)
        
    def log_sum_exp(self, values):
        m = max(values)
        total = 0.0
        for v in values:
            total += math.exp(v - m)
        return m + math.log(total)
        
    def LogGaussianKernel(self, u):
        return -0.5 * math.log(2 * math.pi) -0.5 * u * u


if __name__ == "__main__":
    bandwitdth_values = [0.005, 0.03, 0.2, 0.8, 3.2]

    for bandwidth in bandwitdth_values:
        train_acc = []
        test_acc = []
        for seed in range(43, 48):
            inputs, outputs = make_dataset(1500, random_state=seed)
            inputs_train, outputs_train = inputs[:1200], outputs[:1200]
            inputs_test, outputs_test = inputs[1200:], outputs[1200:]

            naive_bayes_classifier = NaiveBayesClassifier(bandwidth=bandwidth)
            naive_bayes_classifier.fit(inputs_train, outputs_train)

            train_pred = naive_bayes_classifier.predict(inputs_train)
            test_pred = naive_bayes_classifier.predict(inputs_test)
            train_acc.append(np.mean(train_pred == outputs_train))
            test_acc.append(np.mean(test_pred == outputs_test))


        file_name = f"naive_bayes_results/bandwidth_{bandwidth}"
        plot_boundary(file_name, naive_bayes_classifier, inputs_test, outputs_test, title=f"bandwidth={bandwidth}")
        print(f"h = {bandwidth}, test = {np.mean(test_acc):.3f} +/- {np.std(test_acc):.3f}, train = {np.mean(train_acc):.3f} +/- {np.std(train_acc):.3f}")