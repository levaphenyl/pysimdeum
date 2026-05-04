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


@pytest.fixture(scope='module')
def nl_stats():
    """Load NL statistics once for all tests in this module."""
    return Statistics('NL')


class TestParentEndUseClass:
    """The class methods inherited from EndUse work as intended."""

    @pytest.mark.parametrize('offset, expected', [
        pytest.param(0, pd.Timedelta(), id='null_offset'),
        pytest.param('2 Hours', pd.Timedelta(hours=2), id='hours_offset'),
        pytest.param('20 Minutes', pd.Timedelta(minutes=2), id='minutes_offset'),
    ])
    def test_construction_and_offset(self, offset, expected):
        """Instantiation sets the offset."""
        end_use = EndUse(statistics={'offset': offset})
        assert end_use.offset == expected

    def test_init_consumption(self):
        """init_consumption creates a zero-filled DataFrame with correct shape."""
        end_use = EndUse(statistics={'offset': 100})
        # Omitting users raises an exception.
        with pytest.raises(Exception):
            end_use.init_consumption()

        users = ['a', 'b', 'c']
        df = end_use.init_consumption(users=users)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (N_PTS, len(users))
        assert (df == 0).all().all()
        assert df.name == 'EndUse'
        assert list(df.columns) == ['user_1', 'user_2', 'user_3']

    def test_probability_distribution(self):
        """Base usage_probability returns a valid normalised distribution."""
        prob = EndUse.usage_probability() 
        assert isinstance(prob, pd.Series)
        assert len(prob) == N_PTS
        assert prob.sum() == pytest.approx(1.0)
        assert (prob >= 0).all()


class TestChildEndUseClasses:
    """All end-use classes construct from NL statistics without error."""

    @pytest.mark.parametrize('cls, stats_key', ALL_END_USES)
    def test_construction_and_offset(self, nl_stats, cls, stats_key):
        """Instantiation sets a non-negative integer offset."""
        end_use = cls(statistics=nl_stats.end_uses[stats_key])
        assert isinstance(end_use, cls)
        assert isinstance(end_use.offset, int)
        assert end_use.offset == nl_stats.end_uses[stats_key]['offset']

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
