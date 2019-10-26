import pandas as pd
from sklearn.base import BaseEstimator


class Uplift(BaseEstimator):

    def __init__(self, estimator, searchcv, **searchargs):
        self._treated = searchcv(estimator=estimator, **searchargs)
        self._untreated = searchcv(estimator=estimator, **searchargs)


    def fit(self, X, y, **params):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def cate(self, X):
        raise NotImplementedError()