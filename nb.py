"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from data import make_dataset
from plot import plot_boundary

class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    """Naive Bayes classifier."""

    def __init__(self, bandwidth=0.2):
        self.bandwidth = bandwidth

    def _gaussian_kernel(self, u): # can return somethin smaller then the epsilon machin -> return zero
        return (1 / np.sqrt(2 * np.pi)) * np.exp((-u**2)/2) # near the 0 if very negative

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

        y = np.asarray(y) # classes
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        # ====================
        # TODO your code here (KDE Naive Bayes).
        # ====================

        # class identification
        self.classes_ = np.unique(y) # Returns the sorted unique elements of an array. There are three optional outputs in addition to the unique elements:
        
        # log prior computation
        # log prior -> proportion that a given points is in the class C -> the number of point in the class divided 
        # by the number total of points

        self.x_per_c_ = []
        self.log_priors_ = []
        for c in self.classes_:
            # exemple: y == [0, 1, 1, 0], 
            # when c == 1 -> y == [false, true, true, false] 
            # this mask is applied to X
            X_c = X[y == c]
            self.x_per_c_.append(X_c)
            self.log_priors_.append(np.log(X_c.shape[0]/X.shape[0])) # shape = tuple wich contain the dimension of the matrix X_c

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

        # ====================
        # TODO your code here.
        # ====================

        log_prob = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_prob, axis=1)]

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

        # ====================
        # TODO your code here.
        # ====================

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

        # ====================
        # TODO your code here.
        # ====================

        # Numerator


        # Building plan
        # Loop on inputs
        # Loop on classes
            # compute normalization coef
            # pull training points of the current class
            # Loop on each feature
                # build an array for the input of _gaussian_kernel
                # sum all the components
                # normalize the sum
                # compute the log
            # place the result in the corresponding column of the result matrix

        X = np.asarray(X)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        log_prob = np.zeros((n_samples, n_classes))

        for x in range(n_samples):
            for j in range(n_classes):
                score = self.log_priors_[j]

                X_c = self.x_per_c_[j]
                n_y = X_c.shape[0]
                norm = 1 / (n_y * self.bandwidth)

                for f in range(n_features):
                    u = (X[x, f] - X_c[:, f]) / self.bandwidth # : mean all the elements
                    k = self._gaussian_kernel(u)
                    density_input = norm * np.sum(k) # np.sum(k) can be 0.0
                    score += np.log(density_input + 1e-300) # add an epsilon to avoid log of 0

                log_prob[x, j] = score

        # Problem:
        # if log_prob << 0 -> np.exp(log_prob) is rounded to 0.0 -> np.log(0.0) == -inf

        # Exponent property:
        # log_prob == m + (log_prob - m)
        # => exp(log_prob) == exp(m + (log_prob - m))
        # => exp(log_prob) == exp(m) * exp(log_prob - m)
        # => log(exp(log_prob)) == log(exp(m) * exp(log_prob - m))
        # => log_prob == m + log(exp(log_prob - m))
        # -> the principle stays identical even with np.sum

        # if m == max, all the terms (log_prob - m) are <= 0 and the biggest one is == 0,
        # so at least one term is np.exp(0) == 1
        # -> the sum is always >= 1
        # -> prevents the case where the sum is very close to 0 and gives log(0)
        m = np.max(log_prob, axis = 1, keepdims=True)
        log_p_x = m + np.log(np.sum(np.exp(log_prob - m), axis=1, keepdims=True))
        return log_prob - log_p_x

if __name__ == "__main__":

    bandwidths = [0.005, 0.03, 0.2, 0.8, 3.2]

    for h in bandwidths:
        train_acc, test_acc = [], []
        for seed in range(5):
            inputs, outputs = make_dataset(1500, random_state=seed)
            inputs_train, outputs_train = inputs[:1200], outputs[:1200]
            inputs_test, outputs_test = inputs[1200:], outputs[1200:]

            nbc = NaiveBayesClassifier(bandwidth=h)
            nbc.fit(inputs_train, outputs_train) 

            train_acc.append(nbc.score(inputs_train, outputs_train))
            test_acc.append(nbc.score(inputs_test, outputs_test))


            if seed == 0:
                plot_boundary(f"plot_boundary_nbc/pb_{h}", nbc, inputs_test, outputs_test, mesh_step_size=0.1,  title=f"Naive Bayes (h = {h})")

        mean_test = np.mean(test_acc)
        std_test = np.std(test_acc) 

        print("  h =", h)
        print("  test accuracy :", mean_test, "+/-", std_test)

