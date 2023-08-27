import numpy as np


class RandomForestClassifierTensorFlow:
    def __init__(self, in_forest):
        self.forest = in_forest

    def fit(self, X, y, verbose=None):
        return self.forest.fit(X, np.array(y).ravel())

    def evaluate(self, X, y, verbose=None):
        score = self.forest.score(X, np.array(y).ravel())
        return [score] * 2

    def predict(self, X, verbose=None):
        return self.forest.predict(X)
