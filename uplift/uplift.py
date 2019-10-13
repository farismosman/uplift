import pandas as pd
from sklearn.base import BaseEstimator


class Uplift(BaseEstimator):

    def __init__(self, estimator, searchcv, propensity_column, treatment_column, outcome_column, **searchargs):
        self._treated = searchcv(estimator=estimator, **searchargs)
        self._untreated = searchcv(estimator=estimator, **searchargs)
        self.propensity_column = propensity_column
        self.treatment_column = treatment_column
        self.outcome_column = outcome_column


    def fit(self, X, y, **params):
        x_treated, x_untreated, y_treated, y_untreated = self.split_treated_untrated(X, y)
        self._treated = self._treated.fit(x_treated, y_treated, params)
        self._untreated = self._untreated.fit(x_untreated, y_untreated, params)

        return self

    def predict_proba(self, X):
        pass

    def cate(self, X):
        pass
    
    def split_treated_untrated(self, X, y):
        treated = X[X[self.treatment_column] == 1].drop(columns=self.treatment_column)
        untreated = X[X[self.treatment_column] == 0].drop(columns=self.treatment_column)

        y_treated = y.loc[treated.index]
        y_untreated = y.loc[untreated.index]

        return treated, untreated, y_treated, y_untreated