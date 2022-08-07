# This is created by Paul
import copy
import numpy as np
from collections import Counter

def accuracy(y_true, y_pred):
    acc = np.sum(y_true==y_pred)/len(y_true)
    return acc

class KNN_scratch(object):

    def __init__(self, k=3):
        self.k = k

    @staticmethod
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum(x1 - x2) ** 2)

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self,x):
        # Compute distance
        distance = [self.euclidean_distance(x, X_train) for X_train in self.X_train]

        # Get k nearest samples, labels
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Majority vote, get most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class NaiveBayes_scratch(object):

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, variance, prior
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0]/float(n_samples) # Calculate the prior probability

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = self._priors[idx]
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator

# Gradient descent for linear regression
class LinearRegression_GD(object):

    def __init__(self, n_iters = 1500, lr=0.01):
        self.n_iters = n_iters
        self.lr = lr
        self.dj_dw = 0
        self.dj_db = 0
        self.weight = 0
        self.bias = 0

    def _compute_gradient(self, X, y):
        m = X.shape[0]

        for i in range(m):
            f_wb = np.dot(X[i], self.weight) + self.bias
            err = f_wb - y[i]
            self.dj_dw = self.dj_dw + err*X[i]
            self.dj_db = self.dj_db + err
        self.dj_dw = self.dj_dw/m
        self.dj_db = self.dj_db/m
        return None

    def fit(self, X, y):

        for _ in range(self.n_iters):
            self._compute_gradient(X, y)
            self.weight = self.weight - self.lr*self.dj_dw
            self.bias = self.bias - self.lr*self.dj_db
        return None

    def predict(self, X_pred):
        m = X_pred.shape[0]
        y_pred = np.zeros(m)

        for i in range(m):
            y_pred[i] = np.dot(X_pred[i], self.weight) +self.bias
        return y_pred

# Normal Equation for linear regression
class LinearRegression_NE(object):

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape
        X_b = np.c_[np.ones((m,n)), X]
        self.theta = np.linalg.inv(np.dot(X_b.T, X_b)).dot(X_b.T).dot(y)
        return None

    def predict(self,X_pred):
        m, n = X_pred.shape
        X_pred_b = np.c_[np.ones((m,n)), X_pred]
        y_pred = np.dot(X_pred_b, self.theta)

        return y_pred


class LogisticRegression_scratch(object):

    def __init__(self, n_iters = 1500, lr=0.01):
        self.n_iters = n_iters
        self.lr = lr
        self.weight = None
        self.bias = 0

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weight = np.zeros(n)

        for _ in range(self.n_iters):
            f_wb = self.sigmoid(np.dot(X, self.weight)+self.bias)

            dw = np.dot(X.T, (f_wb-y))/m
            db = np.sum(f_wb-y)/m

            self.weight -= self.lr*dw
            self.bias -= self.lr*db
        return None

    def predict(self, X):

        f_wb = self.sigmoid(np.dot(X, self.weight) + self.bias)
        y_pred = np.array([1 if i >0.5 else 0 for i in f_wb])
        return y_pred

class DecisionTree_scratch(object):
    pass
