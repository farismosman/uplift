import pandas as pd
import numpy as np
from unittest import TestCase
from uplift.utils import sample_weights

from numpy.testing import assert_allclose


class UtilsTest(TestCase):

    def test_sample_weights_should_compute_weights_for_treated_samples(self):
        _treatments = pd.Series(data=[1, 1, 1], name='treatment')
        _propensities = pd.Series(data=[0.5, 0.7, 0.8], name='propensity')

        weights = sample_weights(treatments=_treatments, propensities=_propensities)

        assert_allclose(weights, np.array([2, 3.33, 5]), atol=1e-2)

    def test_sample_weights_should_compute_weights_for_untreated_samples(self):
        _treatments = pd.Series(data=[0, 0, 0], name='treatment')
        _propensities = pd.Series(data=[0.1, 0.3, 0.2], name='propensity')

        weights = sample_weights(treatments=_treatments, propensities=_propensities)

        assert_allclose(weights, np.array([10, 3.33, 5]), atol=1e-2)

    def test_sample_weights_should_cap_propensities_for_untreated_samples(self):
        _treatments = pd.Series(data=[0, 0, 0], name='treatment')
        _propensities = pd.Series(data=[0.0, 0.3, 1.0], name='propensity')

        weights = sample_weights(treatments=_treatments, propensities=_propensities)

        assert_allclose(weights, np.array([20, 3.33, 1.05]), atol=1e-2)

    def test_sample_weights_should_cap_propensities_for_treated_samples(self):
        _treatments = pd.Series(data=[1, 1, 1], name='treatment')
        _propensities = pd.Series(data=[1, 0.99, 0.0], name='propensity')

        weights = sample_weights(treatments=_treatments, propensities=_propensities)

        assert_allclose(weights, np.array([20, 100, 1.05]), atol=1e-2)