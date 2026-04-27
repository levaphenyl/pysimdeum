import pytest
import numpy as np
import pandas as pd

from pysimdeum.utils.patterns import (
    sample_start_time,
    complex_daily_pattern,
    complex_enduse_pattern,
    complex_discharge_pattern,
)

N_PTS = 24 * 60 * 60  # One point per second.
HOURLY_PATTERN = [
    34, 24, 17, 43, 41, 57, 57, 212, 290, 338, 392, 325, 226, 222, 200, 175, 124, 106, 139, 185,
    133, 165, 101, 147, 34
]
HOURLY_CONFIG = {
    'daily_pattern_input': {
        'x': ' '.join(map(str, HOURLY_PATTERN)),
    }
}
QUARTER_PATTERN = [
    0.0004, 0.0002, 0.0002, 0.0001, 0.0001, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0001, 0.0001,
    0.0002, 0., 0.0004, 0.0011, 0.0015, 0.0011, 0., 0.0041, 0.0072, 0.0096, 0.018, 0.018, 0.0227,
    0.0229, 0.0221, 0.0155, 0.0153, 0.0155, 0.0139, 0.0091, 0.0087, 0.0068, 0.006, 0.0038, 0.0043,
    0.0042, 0.0063, 0.0061, 0.0115, 0.0158, 0.0326, 0.0214, 0.0251, 0.023, 0.0237, 0.0154, 0.0117,
    0.0072, 0.0059, 0.0037, 0.0038, 0.0029, 0.004, 0.0031, 0.0044, 0.0059, 0.0131, 0.0121, 0.0214,
    0.0239, 0.0453, 0.0348, 0.0462, 0.0423, 0.059, 0.0393, 0.0427, 0.0336, 0.0311, 0.0176, 0.0159,
    0.0104, 0.0075, 0.004, 0.0049, 0.0033, 0.0023, 0.0024, 0.0024, 0.0027, 0.0032, 0.0022, 0.0026,
    0.0018, 0.0021, 0.0013, 0.0012, 0.0005, 0., 0.0004
]
QUARTER_CONFIG = {
    'daily_pattern_input': {
        'x': ' '.join(map(str, QUARTER_PATTERN))
    }
}
DAILY_PATTERN_CASES = [
    pytest.param(HOURLY_PATTERN, HOURLY_CONFIG,  '1h',    id='hourly'),
    pytest.param(QUARTER_PATTERN, QUARTER_CONFIG, '15Min', id='15min'),
]
ENDUSE_CONFIG = {
    'enduse_pattern_input': {
        'intensity': 0.1667,
        'runtime': 7200,
        'cycle_times': [
            {'start': 0,    'end': 121},
            {'start': 3600, 'end': 3660},
            {'start': 4920, 'end': 4980},
            {'start': 6120, 'end': 6180},
        ],
    },
    'discharge_pattern_input': {
        'discharge_time': 60,
    },
}


@pytest.fixture
def uniform_prob_joint():
    """A uniform probability distribution for one day."""
    return np.full(N_PTS, 1/N_PTS)


@pytest.fixture
def peak_prob_joint():
    """A probability distribution with a single peak for one day."""
    prob_joint = np.zeros(N_PTS)
    peak_radius = 30
    peak_start = N_PTS // 2 - peak_radius
    peak_end = N_PTS // 2 + peak_radius
    prob_joint[peak_start:peak_end] = 1.
    return prob_joint / prob_joint.sum()


class TestSampleStartTime:
    @pytest.mark.parametrize("prob_joint_type", ["uniform", "peak"])
    @pytest.mark.parametrize("day_num", [0, 1])
    @pytest.mark.parametrize("duration", [60, 600])
    @pytest.mark.parametrize("previous_events", [
        [],  # No previous events.
        [(0, 10 * 60 * 60)],  # One previous event covering the first 10 hours of the first day.
    ])
    @pytest.mark.parametrize("offset", [0, 7200])
    def test_sample_start_time(self, prob_joint_type, day_num, duration, previous_events, offset, uniform_prob_joint, peak_prob_joint):
        """Tests the sample_start_time function under normal use."""
        peak_radius = 30
        if prob_joint_type == "uniform":
            prob_joint = uniform_prob_joint
        else:
            prob_joint = peak_prob_joint

        start, end = sample_start_time(prob_joint, day_num, duration, previous_events, offset)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert (end - start) == duration
        assert start >= day_num * N_PTS
        # Test that the probability distribution is respected
        if prob_joint_type == "peak":
            assert day_num * N_PTS + N_PTS // 2 - peak_radius <= start <= day_num * N_PTS + N_PTS // 2 + peak_radius

        # Test that the sampled time does not overlap with any previous events
        for event_start, event_end in previous_events:
            assert not (
                (event_start <= start < event_end + offset)
                or (start < event_start <= start + int(duration) + offset)
            ), f"Sampled start {start}, end {end} overlaps with previous event ({event_start}, {event_end}) with offset {offset}"


    def test_sample_start_time_empty_prob_joint(self):
        """Test with an empty prob_joint, expecting an error from np.random.choice."""
        prob_joint = np.array([])
        day_num = 0
        duration = 60
        previous_events = []
        with pytest.raises(ValueError, match="a must be greater than 0"):
            sample_start_time(prob_joint, day_num, duration, previous_events)


    def test_sample_start_time_inf_loop(self, peak_prob_joint):
        """Test against infinite loops when all conditions are invalid."""
        with pytest.raises(RuntimeError, match="Could not find a valid start time"):
            previous_events = [(0, 24 * 60 * 60)]  # One previous event covering the full day.
            sample_start_time(
                prob_joint=peak_prob_joint,
                day_num=0,
                duration=60,
                previous_events=previous_events
            )


class TestComplexDailyPattern:
    """Behavioural tests for complex_daily_pattern."""

    @pytest.mark.parametrize('pattern,config,freq', DAILY_PATTERN_CASES)
    def test_output_structure(self, pattern, config, freq):
        """Output is an 86400-point TimedeltaIndex Series with no NaNs."""
        result = complex_daily_pattern(config, freq=freq)
        assert isinstance(result, pd.Series)
        assert isinstance(result.index, pd.TimedeltaIndex)
        assert len(result) == N_PTS
        assert not result.isna().any()

    @pytest.mark.parametrize('pattern,config,freq', DAILY_PATTERN_CASES)
    def test_output_values(self, pattern, config, freq):
        """All values are finite and non-negative after resampling and interpolation."""
        result = complex_daily_pattern(config, freq=freq)
        assert (result >= 0).all()
        assert np.isfinite(result.values).all()
        assert result.max() <= max(pattern)
        assert result.min() >= min(pattern)
        assert all(x in result.values for x in pattern)

    @pytest.mark.parametrize('bad_freq', ['30Min', '2h', '10s', 'D'])
    def test_invalid_freq_raises(self, bad_freq):
        """Unsupported freq values raise ValueError."""
        with pytest.raises(ValueError, match='freq'):
            complex_daily_pattern(HOURLY_CONFIG, freq=bad_freq)


class TestComplexEndusePattern:
    """Behavioural tests for complex_enduse_pattern."""

    def test_output_structure(self):
        """Output is a runtime-length TimedeltaIndex Series with no NaNs."""
        enduse_pat = complex_enduse_pattern(ENDUSE_CONFIG)
        runtime = ENDUSE_CONFIG['enduse_pattern_input']['runtime']
        assert isinstance(enduse_pat, pd.Series)
        assert isinstance(enduse_pat.index, pd.TimedeltaIndex)
        assert len(enduse_pat) == runtime
        assert not enduse_pat.isna().any()

    def test_only_two_values(self):
        """Series contains exactly two distinct values: 0.0 and the configured intensity."""
        enduse_pat = complex_enduse_pattern(ENDUSE_CONFIG)
        intensity = ENDUSE_CONFIG['enduse_pattern_input']['intensity']
        unique = set(np.round(enduse_pat.unique(), 4))
        assert unique == {0.0, round(intensity, 4)}

    def test_cycle_values(self):
        """Correct intensity or zero at known positions relative to cycle windows."""
        enduse_pat = complex_enduse_pattern(ENDUSE_CONFIG)
        idx_expected = [
            (0,    0.1667),  # start of cycle 1
            (100,  0.1667),  # inside cycle 1
            (200,  0.0),     # after cycle 1 end (121), before cycle 2
            (3600, 0.1667),  # start of cycle 2
            (3630, 0.1667),  # inside cycle 2
            (3700, 0.0),     # after cycle 2 end (3660)
            (4920, 0.1667),  # start of cycle 3
            (6120, 0.1667),  # start of cycle 4
            (6200, 0.0),     # after cycle 4 end (6180)
        ]
        assert all(enduse_pat.iloc[i] == pytest.approx(x, abs=1e-4) for i, x in idx_expected)


@pytest.fixture(scope='module')
def end_use_discharge_pat():
    """Real discharge pattern from ENDUSE_CONFIG."""
    eup = complex_enduse_pattern(ENDUSE_CONFIG)
    return eup, complex_discharge_pattern(ENDUSE_CONFIG, eup)


class TestComplexDischargePattern:
    """Behavioural tests for complex_discharge_pattern."""

    def test_output_structure(self, end_use_discharge_pat):
        """Output is a runtime-length TimedeltaIndex Series with no NaNs."""
        discharge_pat = end_use_discharge_pat[1]
        runtime = ENDUSE_CONFIG['enduse_pattern_input']['runtime']
        assert isinstance(discharge_pat, pd.Series)
        assert isinstance(discharge_pat.index, pd.TimedeltaIndex)
        assert len(discharge_pat) == runtime
        assert not discharge_pat.isna().any()

    def test_discharge_occurs_and_is_valid(self, end_use_discharge_pat):
        """Discharge events exist, are all non-negative, and sum to a positive total."""
        discharge_pat = end_use_discharge_pat[1]
        assert (discharge_pat >= 0).all()
        assert (discharge_pat > 0).any()
        assert discharge_pat.sum() > 0

    def test_no_overlap_with_enduse(self, end_use_discharge_pat):
        """Discharge never fires simultaneously with active end-use consumption."""
        enduse_pat, discharge_pat = end_use_discharge_pat
        overlap = (enduse_pat > 0) & (discharge_pat > 0)
        assert not overlap.any(), (
            f'Found {overlap.sum()} time steps where discharge overlaps active consumption'
        )

    def test_water_conservation(self, end_use_discharge_pat):
        """Total discharge volume equals total consumed volume within 5% tolerance."""
        enduse_pat, discharge_pat = end_use_discharge_pat
        total_consumed = enduse_pat.sum()
        total_discharged = discharge_pat.sum()
        assert total_discharged == pytest.approx(total_consumed, rel=0.05)
