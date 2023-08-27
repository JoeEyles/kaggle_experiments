import numpy as np


class SVMClassifierTensorFlow:
    def __init__(self, in_svm):
        self.svm = in_svm

    def fit(self, X, y, verbose=None):
        return self.svm.fit(X, np.array(y).ravel())

    def evaluate(self, X, y, verbose=None):
        score = self.svm.score(X, np.array(y).ravel())
        return [score] * 2

    def predict(self, X, verbose=None):
        return self.svm.predict(X)
