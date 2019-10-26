from uplift.uplift import Uplift


class TwoModelsUplift(Uplift):

    def fit(self, X, y, treatment_column, **params):
        xtreated, xuntreated, ytreated, yuntreated = self.split_treated_untrated(X, y, treatment_column)
        self._treated = self._treated.fit(xtreated, ytreated)
        self._untreated = self._untreated.fit(xuntreated, yuntreated)

        return self


    def split_treated_untrated(self, X, y, treatment_column):
        xtreated = X[X[treatment_column] == 1].drop(columns=treatment_column)
        xuntreated = X[X[treatment_column] == 0].drop(columns=treatment_column)

        ytreated = y.loc[xtreated.index]
        yuntreated = y.loc[xuntreated.index]

        return xtreated, xuntreated, ytreated, yuntreated