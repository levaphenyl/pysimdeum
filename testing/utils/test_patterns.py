import pytest
import numpy as np

from pysimdeum.utils.patterns import (
    sample_start_time,
)

N_PTS = 24 * 60 * 60  # One point per second.


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


@pytest.mark.parametrize("prob_joint_type", ["uniform", "peak"])
@pytest.mark.parametrize("day_num", [0, 1])
@pytest.mark.parametrize("duration", [60, 600])
@pytest.mark.parametrize("previous_events", [
    [],  # No previous events.
    [(0, 10 * 60 * 60)],  # One previous event covering the first 10 hours of the first day.
])
@pytest.mark.parametrize("offset", [0, 7200])
def test_sample_start_time(prob_joint_type, day_num, duration, previous_events, offset, uniform_prob_joint, peak_prob_joint):
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


def test_sample_start_time_empty_prob_joint():
    """Test with an empty prob_joint, expecting an error from np.random.choice."""
    prob_joint = np.array([])
    day_num = 0
    duration = 60
    previous_events = []

    with pytest.raises(ValueError, match="a must be greater than 0"):
        sample_start_time(prob_joint, day_num, duration, previous_events)
