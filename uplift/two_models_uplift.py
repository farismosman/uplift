from uplift.uplift import Uplift
from uplift.utils import sample_weights


class TwoModelsUplift(Uplift):

    def fit(self, X, y, **params):
        treatment_column = params['treatment_column']
        propensity_column = params['propensity_column']

        treatments = X[treatment_column]
        propensity = X.pop(propensity_column)
        weights = sample_weights(treatments, propensity)

        xtreated, xuntreated, ytreated, yuntreated = self.split_treated_untrated(X, y, treatment_column)
        treated_weights = weights.loc[xtreated.index]
        untreated_weights = weights.loc[xuntreated.index]

        self._treated = self._treated.fit(xtreated, ytreated, sample_weight=treated_weights)
        self._untreated = self._untreated.fit(xuntreated, yuntreated, sample_weight=untreated_weights)

        return self

    def predict_proba(self, X):
        p_treated = self._treated.predict_proba(X)
        p_untreated = self._untreated.predict_proba(X)

        return p_treated - p_untreated


    def split_treated_untrated(self, X, y, treatment_column):
        xtreated = X[X[treatment_column] == 1].drop(columns=treatment_column)
        xuntreated = X[X[treatment_column] == 0].drop(columns=treatment_column)

        ytreated = y.loc[xtreated.index]
        yuntreated = y.loc[xuntreated.index]

        return xtreated, xuntreated, ytreated, yuntreated