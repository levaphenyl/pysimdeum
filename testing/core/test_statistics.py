"""
Integration tests for the Statistics dataclass (pysimdeum.core.statistics).

These tests use the real NL and UK data files that are shipped with the
pysimdeum package — no mocking of I/O, toml.load, or pattern helpers.
"""
import os
import pytest
from unittest.mock import patch

import toml
from pysimdeum.core.statistics import Statistics
from pysimdeum.data import DATA_DIR


# Constants derived from the real NL data layout
EXPECTED_END_USE_KEYS = [
    'Wc', 'Bathtub', 'BathroomTap', 'Dishwasher',
    'KitchenTap', 'OutsideTap', 'Shower', 'WashingMachine',
]
EXPECTED_FULL_PATTERN_END_USES = ['WashingMachine', 'Dishwasher']
PATTERN_KEYS_FULL = ['daily_pattern', 'enduse_pattern', 'discharge_pattern']


# Module-scoped fixture — built once, shared across all tests
@pytest.fixture(scope='module')
def nl_stats():
    """Construct Statistics('NL') once using real NL data files."""
    return Statistics(country='NL')


def test_statistics_default_instanciation():
    """Smoke tests: Statistics() constructs without error."""
    stats = Statistics()
    assert stats is not None
    assert stats.statisticsdir.startswith(DATA_DIR)
    assert isinstance(stats, Statistics)

@pytest.mark.parametrize('country', ['NL', 'UK'])
def test_statistics_country_instanciation(country):
    stats = Statistics(country=country)
    assert stats.country == country
    assert stats.statisticsdir.startswith(DATA_DIR)
    assert country in stats.statisticsdir
    assert os.path.isdir(stats.statisticsdir)


class TestStatisticsCustomDir:
    """Integration test: Statistics accepts a real directory path instead of a country code."""

    def test_custom_dir_mode(self, nl_stats):
        """Passing a real directory path loads data correctly and sets country to None."""
        real_dir = nl_stats.statisticsdir
        stats = Statistics(country=real_dir)
        assert stats.country is None
        assert stats.statisticsdir == real_dir
        assert len(stats.end_uses) == len(EXPECTED_END_USE_KEYS)


class TestStatisticsHousehold:
    """Tests for the household dict loaded from household_statistics.toml."""

    def test_household_is_dict(self, nl_stats):
        assert isinstance(nl_stats.household, dict)
        assert len(nl_stats.household) > 0
        assert all(isinstance(k, str) for k in nl_stats.household)


class TestStatisticsDiurnalPattern:
    """Tests for the diurnal_pattern dict loaded from diurnal_patterns.toml."""

    def test_diurnal_pattern_is_dict(self, nl_stats):
        assert isinstance(nl_stats.diurnal_pattern, dict)
        assert len(nl_stats.diurnal_pattern) > 0
        assert all(isinstance(k, str) for k in nl_stats.diurnal_pattern)


class TestStatisticsEndUses:
    """Tests for the end_uses dict populated from 8 TOML files."""

    def test_end_uses_is_dict(self, nl_stats):
        assert isinstance(nl_stats.end_uses, dict)
        assert len(nl_stats.end_uses) == len(EXPECTED_END_USE_KEYS)
        for k in EXPECTED_END_USE_KEYS:
            assert k in nl_stats.end_uses, f'{k!r} missing from end_uses'  # k!r equivalent to repr(k)
            assert isinstance(nl_stats.end_uses[k], dict)
            assert len(nl_stats.end_uses[k]) > 0
            assert 'frequency' in nl_stats.end_uses[k]
            assert 'offset' in nl_stats.end_uses[k]


class TestStatisticsPatterns:
    """Tests correct pattern initialization."""

    @pytest.mark.parametrize('enduse', EXPECTED_FULL_PATTERN_END_USES)
    @pytest.mark.parametrize('pattern', PATTERN_KEYS_FULL)
    def test_washing_machine_has_daily_pattern(self, nl_stats, enduse, pattern):
        """Example: check that WashingMachine has daily_pattern."""
        assert pattern in nl_stats.end_uses[enduse]
        assert nl_stats.end_uses[enduse][pattern] is not None

    def test_kitchen_tap_has_daily_pattern(self, nl_stats):
        """KitchenTap has daily_pattern."""
        assert 'daily_pattern' in nl_stats.end_uses['KitchenTap']
        assert nl_stats.end_uses['KitchenTap']['daily_pattern'] is not None


class TestStatisticsConvertToDict:
    """Unit tests for _convert_to_dict with __post_init__ bypassed, no real data needed."""

    def _instance(self):
        """Return a Statistics instance with __post_init__ disabled."""
        with patch.object(Statistics, '__post_init__', lambda self: None):
            return Statistics()

    def test_plain_dict(self):
        """Plain dict and TOML dict pass through."""
        expected = {'a': 1}
        assert self._instance()._convert_to_dict(expected) == expected
        toml_dict = toml.decoder.TomlDecoder().get_empty_inline_table()
        toml_dict['a'] = 1
        assert self._instance()._convert_to_dict(toml_dict) == expected

    def test_nested_dict(self):
        """Nested dict and TOML dict are recursed."""
        expected = {'x': {'y': 2}}
        assert self._instance()._convert_to_dict(expected) == expected
        nested_toml_dict = {'x': toml.decoder.TomlDecoder().get_empty_inline_table()}
        nested_toml_dict['x']['y'] = 2
        assert self._instance()._convert_to_dict(nested_toml_dict) == expected

    def test_list_of_ints(self):
        """List of ints passes through."""
        assert self._instance()._convert_to_dict([1, 2, 3]) == [1, 2, 3]

    def test_list_of_dicts(self):
        """List of dicts is recursed."""
        assert self._instance()._convert_to_dict([{'a': 1}]) == [{'a': 1}]

    @pytest.mark.parametrize('value', [0, 3.14, 'hello', True, None, False])
    def test_scalar(self, value):
        """Scalar value passes through unchanged."""
        assert self._instance()._convert_to_dict(value) == value

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        assert self._instance()._convert_to_dict({}) == {}

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert self._instance()._convert_to_dict([]) == []