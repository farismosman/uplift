import pandas as pd
from unittest import TestCase
from uplift.uplift import Uplift
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from pandas.util.testing import assert_frame_equal, assert_series_equal



class UpliftTest(TestCase):

    def setUp(self):
        self._uplift = Uplift(
            estimator=LogisticRegression(),
            searchcv=GridSearchCV,
            propensity_column='propensity', 
            treatment_column='treatment',
            outcome_column='outcome', param_grid={
                'penalty' : ['l1', 'l2']                
            })

    def test_split_treated_untrated(self):
        x_treated, x_untreated, y_treated, y_untreated = self._uplift.split_treated_untrated(
            X=pd.DataFrame(
                data={
                    'treatment' : [1, 1, 0, 1, 0, 0],
                    'other': [2, 1, 2, 5, 9, 2]
                    }),
            y=pd.Series(data=[1, 1, 0, 0, 0, 1], name='outcome'))

        assert_frame_equal(x_treated, pd.DataFrame(data={'other' : [2, 1, 5]}, index=[0, 1, 3]))
        assert_frame_equal(x_untreated, pd.DataFrame(data={'other' : [2, 9, 2]}, index=[2, 4, 5]))
        assert_series_equal(y_treated, pd.Series(data=[1, 1, 0], name='outcome', index=[0, 1, 3]))
        assert_series_equal(y_untreated, pd.Series(data=[0, 0, 1], name='outcome', index=[2, 4, 5]))