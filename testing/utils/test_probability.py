"""Unit tests for the probability utility functions (pysimdeum.utils.probability)"""
import pytest
import numpy as np
from pysimdeum.utils.probability import (
    sample_value
)


VALID_CASES = [
    pytest.param('Poisson',           {'average': 2.5},               0,      6,  id='poisson-average'),
    pytest.param('Poisson',           {'average': 0.80},              0,      2,  id='poisson-average-low'),
    pytest.param('Poisson',           {'lam': 0.80},                  0,      4,  id='poisson-lam'),
    pytest.param('Lognormal',         {'average': '57 Seconds'},      0,   6000,  id='lognormal-str-avg'),
    pytest.param('Lognormal',         {'average': 14, 'sigma': 0.5},  0,    600,  id='lognormal-avg-sigma'),
    pytest.param('Uniform',           {'low': 0., 'high': 0.167},     0,  0.167,  id='uniform-small'),
    pytest.param('Uniform',           {'low': 20, 'high': 40},       20,     40,  id='uniform-int-range'),
    pytest.param('Negative_binomial', {'average': 10.1, 'sigma': 7},  0,     71,  id='nb-average-sigma'),
    pytest.param('Negative_binomial', {'n': 2.622, 'p': 0.206},       0,     71,  id='nb-n-p'),
    pytest.param('Binomial',          {'n': 1, 'p': 0.69},            0,      1,  id='binomial'),
    pytest.param('Chisquare',         {'df': '9.3 Minutes'},          0,   5160,  id='chisquare-str-df'),
]

INVALID_CASES = [
    pytest.param('Poisson',           {'lambda': 3.8},          id='poisson-wrong-kwarg'),
    pytest.param('Binomial',          {'n': 3},                 id='binomial-missing-p'),
    pytest.param('Binomial',          {'n': 1, 'p': 2.13},      id='binomial-p-gt-1'),
    pytest.param('Binomial',          {'n': 1, 'sigma': 2.13},  id='binomial-wrong-kwarg'),
    pytest.param('very_special_dist', {'average': 29},          id='unsupported-dist'),
]


class TestSampleValue:
    @pytest.mark.parametrize('dist_name, kwargs, lower, upper', VALID_CASES)
    def test_valid_config_returns_value_in_range(self, dist_name, kwargs, lower, upper):
        """Valid configs return a numeric scalar within expected bounds."""
        result = sample_value(dist_name, **kwargs)
        assert isinstance(result, (int, float, np.integer, np.floating))
        assert lower <= result <= upper

    @pytest.mark.parametrize('dist_name, kwargs', INVALID_CASES)
    def test_invalid_config_raises_value_error(self, dist_name, kwargs):
        """Invalid or unsupported configs raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            sample_value(dist_name, **kwargs)

    def test_sampling_is_stochastic(self):
        """Repeated calls produce varying results."""
        values = {sample_value('Uniform', low=0, high=1000) for __ in range(20)}
        assert len(values) > 1
