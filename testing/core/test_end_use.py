"""
Tests for pysimdeum.core.end_use using the real NL statistics files.

No mocking — all end-use classes are instantiated with data loaded by
Statistics('NL') from the installed pysimdeum package.
"""

import pytest
import numpy as np
import pandas as pd
from pysimdeum.core.statistics import Statistics
from pysimdeum.core.end_use import (
    EndUse,
    BathroomTap,
    Bathtub,
    Dishwasher,
    KitchenTap,
    OutsideTap,
    NormalShower,
    FancyShower,
    WashingMachine,
    WcNormal,
    WcNormalSave,
    WcNew,
    WcNewSave,
)


N_PTS = 24 * 60 * 60  # One point per second over a day, 86400 points total.
ALL_END_USES = [
    pytest.param(Bathtub,        'Bathtub',        id='Bathtub'),
    pytest.param(BathroomTap,    'BathroomTap',    id='BathroomTap'),
    pytest.param(Dishwasher,     'Dishwasher',     id='Dishwasher'),
    pytest.param(KitchenTap,     'KitchenTap',     id='KitchenTap'),
    pytest.param(OutsideTap,     'OutsideTap',     id='OutsideTap'),
    pytest.param(NormalShower,   'Shower',         id='NormalShower'),
    pytest.param(FancyShower,    'Shower',         id='FancyShower'),
    pytest.param(WashingMachine, 'WashingMachine', id='WashingMachine'),
    pytest.param(WcNormal,       'Wc',             id='WcNormal'),
    pytest.param(WcNormalSave,   'Wc',             id='WcNormalSave'),
    pytest.param(WcNew,          'Wc',             id='WcNew'),
    pytest.param(WcNewSave,      'Wc',             id='WcNewSave'),
]
AGE_GENDER_CONFIG = {
    "distribution": "Poisson",
    "average": {
        "child": {"male": 1.0, "female": 1.5},
        "adult": {"male": 2.0, "female": 2.5},
    },
}
NUMUSERS_CONFIG = {
    "distribution": "Poisson",
    "average": {"1": 0.5, "2": 1.0, "3": 1.5},
}


@pytest.fixture(scope='module')
def nl_stats():
    """Load NL statistics once for all tests in this module."""
    return Statistics('NL')


@pytest.fixture(scope="module")
def end_use():
    """Minimal EndUse (only needs a valid 'offset' to construct)."""
    mock_stats = {
        'offset': '0s',
        'frequency': {
            'distribution': 'Uniform',
            'low': 1,
            'high': 2,
        }
    }
    return EndUse(statistics=mock_stats)


class TestParentEndUseClass:
    """The class methods inherited from EndUse work as intended."""

    @pytest.mark.parametrize('offset, expected', [
        pytest.param(0, 0, id='null_offset'),
        pytest.param('2 Hours', 7200, id='hours_offset'),
        pytest.param('20 Minutes', 1200, id='minutes_offset'),
    ])
    def test_construction_and_offset(self, offset, expected):
        """Instantiation sets the offset."""
        end_use = EndUse(statistics={'offset': offset})
        assert end_use.offset == expected

    def test_init_consumption(self, end_use):
        """init_consumption creates a zero-filled DataFrame with correct shape."""
        # Omitting users raises an exception.
        with pytest.raises(KeyError):
            end_use.init_consumption(users=[])

        users = ['a', 'b', 'c']
        df = end_use.init_consumption(users=users)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (N_PTS, len(users))
        assert (df == 0).all().all()
        assert list(df.columns) == ['user_1', 'user_2', 'user_3']

    def test_probability_distribution(self):
        """Base usage_probability returns a valid normalised distribution."""
        prob = EndUse.usage_probability()
        assert isinstance(prob, pd.Series)
        assert len(prob) == N_PTS
        assert prob.sum() == pytest.approx(1.0)
        assert (prob >= 0).all()

    @pytest.mark.parametrize("dist_config, call_kwargs, exp_name, exp_params", [
        pytest.param({"distribution": "Poisson", "lam": 2.5}, {}, "Poisson", {"lam": 2.5}, id="constant-poisson"),  # Constant (no optional args)
        pytest.param({"distribution": "Uniform", "low": 0, "high": 1}, {}, "Uniform", {"low": 0, "high": 1}, id="constant-uniform"),
        pytest.param({"distribution": "Poisson", "average": {"child": 1.0, "adult": 2.0}}, {"age": "child"}, "Poisson", {"average": 1.0}, id="age-child"),  # Age resolution
        pytest.param(AGE_GENDER_CONFIG, {"age": "child", "gender": "male"}, "Poisson", {"average": 1.0}, id="age-gender-child-male"),  # Age + gender resolution
        pytest.param(NUMUSERS_CONFIG, {"numusers": 1}, "Poisson", {"average": 0.5}, id="numusers-1"),  # Numusers resolution (int -> str lookup)
        pytest.param(NUMUSERS_CONFIG, {"numusers": 3}, "Poisson", {"average": 1.5}, id="numusers-3"),
        pytest.param({"distribution": "Negative_binomial", "n": 1, "p": {"child": 0.3, "adult": 0.5}}, {"age": "child"}, "Negative_binomial", {"n": 1, "p": 0.3}, id="mixed-child"),  # Mixed: scalar params pass through, only dict-valued params are resolved
    ])
    def test_get_statistical_params_success(self, end_use, dist_config, call_kwargs, exp_name, exp_params):
        """All resolution modes return (str, dict) with the correct values."""
        result = end_use.get_statistical_params(dist_config, **call_kwargs)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)
        assert result[0] == exp_name
        assert result[1] == exp_params

    def test_get_statistical_params_no_mutation(self, end_use):
        """The original dist_config dict must not be modified."""
        original = {"distribution": "Poisson", "average": {"child": 1.0}}
        copy = {"distribution": "Poisson", "average": {"child": 1.0}}
        end_use.get_statistical_params(original, age="child")
        assert original == copy

    @pytest.mark.parametrize("dist_config, call_kwargs", [
        pytest.param({}, {}, id="missing-distribution-key"),
        pytest.param(AGE_GENDER_CONFIG, {"age": "nonexistent"}, id="invalid-age"),
        pytest.param(NUMUSERS_CONFIG, {"numusers": 99}, id="invalid-numusers"),
    ])
    def test_get_statistical_params_fail(self, end_use, dist_config, call_kwargs):
        """Missing or mismatched keys raise KeyError."""
        with pytest.raises(KeyError):
            end_use.get_statistical_params(dist_config, **call_kwargs)

    def test_fct_frequency(self, end_use):
        res = {end_use.fct_frequency() for __ in range(10)}
        assert res == {1, 2}


class TestChildEndUseClasses:
    """All end-use classes construct from NL statistics without error."""

    @pytest.mark.parametrize('cls, stats_key', ALL_END_USES)
    def test_construction_and_offset(self, nl_stats, cls, stats_key):
        """Instantiation sets a non-negative integer offset."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        assert isinstance(end_use, cls)
        assert isinstance(end_use.offset, int)
        if isinstance(nl_stats.end_uses[stats_key]['offset'], str):
            assert end_use.offset > 0
        else:
            assert end_use.offset == 0

    @pytest.mark.parametrize('cls, stats_key', [
        pytest.param(BathroomTap, 'BathroomTap', id='BathroomTap'),
        pytest.param(OutsideTap,  'OutsideTap',  id='OutsideTap'),
    ])
    def test_no_arg_frequency(self, nl_stats, cls, stats_key):
        """End-uses whose frequency takes no arguments."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        freq = end_use.fct_frequency()
        assert isinstance(freq, (int, np.integer))
        assert freq >= 0

    @pytest.mark.parametrize('cls, stats_key, age_path', [
        pytest.param(Bathtub,      'Bathtub', ['frequency', 'average'], id='Bathtub'),
        pytest.param(NormalShower, 'Shower',  ['frequency', 'p'],       id='NormalShower'),
        pytest.param(FancyShower,  'Shower',  ['frequency', 'p'],       id='FancyShower'),
    ])
    def test_age_based_frequency(self, nl_stats, cls, stats_key, age_path):
        """End-uses whose frequency depends on user age."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        d = end_use.statistics
        for key in age_path:
            d = d[key]

        for age in d:
            freq = end_use.fct_frequency(age=age)
            assert isinstance(freq, (int, np.integer))
            assert freq >= 0

    @pytest.mark.parametrize('cls, stats_key', [
        pytest.param(Dishwasher,     'Dishwasher',     id='Dishwasher'),
        pytest.param(KitchenTap,     'KitchenTap',     id='KitchenTap'),
        pytest.param(WashingMachine, 'WashingMachine', id='WashingMachine'),
    ])
    def test_numusers_frequency(self, nl_stats, cls, stats_key):
        """End-uses whose frequency depends on the number of household users."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        for n_str in end_use.statistics['frequency']['average']:
            freq = end_use.fct_frequency(numusers=int(n_str))
            assert isinstance(freq, (int, np.integer))
            assert freq >= 0

    @pytest.mark.parametrize('cls', [
        pytest.param(WcNormal,     id='WcNormal'),
        pytest.param(WcNormalSave, id='WcNormalSave'),
        pytest.param(WcNew,        id='WcNew'),
        pytest.param(WcNewSave,    id='WcNewSave'),
    ])
    def test_wc_frequency(self, nl_stats, cls):
        """WC frequency depends on both age and gender."""
        end_use = cls(statistics=nl_stats.end_uses['Wc'])
        age_dict = end_use.statistics['frequency']['average']
        for age in age_dict:
            for gender in age_dict[age]:
                freq = end_use.fct_frequency(age=age, gender=gender)
                assert isinstance(freq, (int, np.integer))
                assert freq >= 0

    def test_bathtub_separate_methods(self, nl_stats):
        """Bathtub exposes duration, intensity, temperature as separate calls."""
        end_use = Bathtub(statistics=nl_stats.end_uses['Bathtub'])
        duration = end_use.fct_duration()
        intensity = end_use.fct_intensity()
        temperature = end_use.temperature()
        assert isinstance(duration, int) and duration > 0
        assert isinstance(intensity, (int, float)) and intensity > 0
        assert isinstance(temperature, (int, float))

    @pytest.mark.parametrize('cls, stats_key', [
        pytest.param(BathroomTap, 'BathroomTap', id='BathroomTap'),
        pytest.param(KitchenTap,  'KitchenTap',  id='KitchenTap'),
        pytest.param(OutsideTap,  'OutsideTap',  id='OutsideTap'),
    ])
    def test_chooser_based(self, nl_stats, cls, stats_key):
        """End-uses that internally pick a subtype via chooser()."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        duration, intensity, temperature = end_use.fct_duration_intensity_temperature()
        assert isinstance(duration, (int, float)) and duration >= 0
        assert isinstance(intensity, (int, float)) and intensity > 0
        assert isinstance(temperature, (int, float))

    @pytest.mark.parametrize('cls', [
        pytest.param(NormalShower, id='NormalShower'),
        pytest.param(FancyShower,  id='FancyShower'),
    ])
    def test_shower_age_based(self, nl_stats, cls):
        """Shower duration/intensity/temperature for every available age key."""
        end_use = cls(statistics=nl_stats.end_uses['Shower'])
        for age in end_use.statistics['duration']['df']:
            duration, intensity, temperature = end_use.fct_duration_intensity_temperature(age=age)
            assert isinstance(duration, (int, float)) and duration >= 0
            assert isinstance(intensity, (int, float)) and intensity > 0
            assert isinstance(temperature, (int, float))

    @pytest.mark.parametrize('cls', [
        pytest.param(WcNormal,     id='WcNormal'),
        pytest.param(WcNormalSave, id='WcNormalSave'),
        pytest.param(WcNew,        id='WcNew'),
        pytest.param(WcNewSave,    id='WcNewSave'),
    ])
    def test_wc(self, nl_stats, cls):
        """WC returns fixed duration, intensity, temperature."""
        end_use = cls(statistics=nl_stats.end_uses['Wc'])
        duration, intensity, temperature = end_use.fct_duration_intensity_temperature()
        assert isinstance(duration, int) and duration > 0
        assert isinstance(intensity, (int, float)) and intensity > 0
        assert isinstance(temperature, (int, float))

    @pytest.mark.parametrize('cls, stats_key', [
        pytest.param(Dishwasher,     'Dishwasher',     id='Dishwasher'),
        pytest.param(WashingMachine, 'WashingMachine', id='WashingMachine'),
    ])
    def test_pattern_shape_and_values(self, nl_stats, cls, stats_key):
        """Dishwasher / WashingMachine return a TimedeltaIndex pattern."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        pattern = end_use.fct_duration_pattern()
        assert isinstance(pattern, pd.Series)
        assert len(pattern) > 0
        assert (pattern >= 0).all()
        assert isinstance(pattern.index, pd.TimedeltaIndex)
