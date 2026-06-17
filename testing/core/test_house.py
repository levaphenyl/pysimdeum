import numpy as np
import pytest
import xarray as xr
from pysimdeum.core.house import Property
from pysimdeum.core.statistics import Statistics
from pysimdeum.core.end_use import EndUse
from statistics import mean


def test_save_and_load_house():
    stats = Statistics()
    prop = Property(statistics=stats)
    house = prop.built_house()
    house.populate_house()
    house.save_house('unittest')

    prop2 = Property(statistics=stats)
    house2 = prop2.built_house(housefile='unittest.house')

    assert house.id == house2.id
    assert house.house_type == house2.house_type


@pytest.fixture(scope="module")
def ready_houses(n_iter=100) -> list:
    """Generate and populate houses in a single expensive function for all tests."""
    houses = []
    stats = Statistics()
    for __ in range(n_iter):
        prop = Property(statistics=stats)
        house = prop.built_house()
        house.populate_house()
        houses.append(house)

    return houses


class TestHouse:
    """Tests for the House class."""

    def test_populate_house(self, ready_houses):
        """Every house has at least one user (occupant).
        The average number of users is 2.29 as per NL statistics.
        """
        number_of_users = [len(x.users) for x in ready_houses]
        assert 0 not in number_of_users
        avg_users = 2.29
        tol = 0.3
        assert avg_users - tol < mean(number_of_users) < avg_users + tol

    def test_furnish_house(self, ready_houses):
        """Test House.furnish_house on multiple instances."""
        wc_names = {'WcNormal', 'WcNormalSave', 'WcNew', 'WcNewSave'}
        bathing_names = {'NormalShower', 'FancyShower', 'Bathtub'}
        for house in ready_houses:
            # After furnishing, the house must have at least one appliance.
            assert hasattr(house, 'appliances')
            assert len(house.appliances) > 0
            # Every appliance must be a subclass of EndUse.
            assert all(isinstance(a, EndUse) for a in house.appliances)
            appliance_unique_names = {a.name for a in house.appliances}
            # Every furnished house should contain a WC variant.
            assert appliance_unique_names & wc_names, (
                f"No WC found among {appliance_unique_names}"
            )
            # Every furnished house should have at least a shower or a bathtub.
            assert appliance_unique_names & bathing_names, (
                f"No shower/bathtub found among {appliance_unique_names}"
            )
            # Each appliance type should appear at most once in a house.
            appliance_names = [a.name for a in house.appliances]
            assert len(appliance_names) == len(appliance_unique_names), (
                f"Duplicate appliances found: {appliance_names}"
            )

    def test_furnish_house_twice(self):
        """Calling furnish_house a second time should not double the appliances."""
        stats = Statistics()
        prop = Property(statistics=stats)
        house = prop.built_house()
        house.populate_house()
        house.furnish_house()
        count_first = len(house.appliances)
        house.furnish_house()
        count_second = len(house.appliances)
        assert count_second == count_first, (
            f"Appliance count changed from {count_first} to {count_second} "
            "after a second call to furnish_house"
        )


    @pytest.mark.parametrize("n_days", [1, 2, 7])
    @pytest.mark.parametrize("num_patterns", [1, 2, 3])
    @pytest.mark.parametrize("simulate_discharge", [True, False])
    def test_simulate_house(self, ready_houses, n_days, num_patterns, simulate_discharge):
        """Test House.simulate on one instance only but with different parameters."""
        house = ready_houses[0]
        expected_dims = {'time', 'user', 'enduse', 'patterns', 'flowtypes'}
        expected_flowtypes = {'hotflow', 'totalflow'}
        # Simulate returns a tuple of consumption and discharge arrays.
        consumption, discharge = house.simulate(duration=f"{n_days} days", num_patterns=num_patterns, simulate_discharge=simulate_discharge)
        # The consumption result must always be an xarray.DataArray.
        assert isinstance(consumption, xr.DataArray)
        # When simulate_discharge=True, discharge must be returned (not None).
        if simulate_discharge:
            assert isinstance(discharge, xr.Dataset)
            discharge = discharge.data_vars['discharge']
        else:
            assert discharge is None

        # Consumption DataArray must have 5 dimensions: time, user, enduse, patterns, flowtypes.
        assert set(consumption.dims) == expected_dims, (
            f"Expected dims {expected_dims}, got {set(consumption.dims)}."
        )
        # Consumption must contain 'totalflow' and 'hotflow' flowtypes.
        flowtypes = set(consumption.coords['flowtypes'].values)
        assert flowtypes == expected_flowtypes, (
            f"Expected flowtypes {expected_flowtypes}, got {flowtypes}."
        )
        # All consumption and discharge values must be >= 0.
        assert np.all(consumption.values >= 0), "Negative consumption values found."
        if discharge is not None:
            assert np.all(discharge.values >= 0), "Negative discharge values found."

        # The patterns dimension must match the num_patterns argument.
        assert consumption.sizes['patterns'] == num_patterns, (
            f"Expected {num_patterns} patterns, got {consumption.sizes['patterns']}"
        )
        # The length matches the duration at 1-sec resolution.
        n_pts = n_days * 24 * 60 * 60
        assert n_pts <= consumption.shape[0] <= n_pts + 1
        if discharge is not None:
            assert n_pts <= discharge.shape[0] <= n_pts + 1

        # The enduse coordinate must include every appliance classname.
        enduse_coords = set(consumption.coords['enduse'].values)
        appliance_classnames = {a.statistics['classname'] for a in house.appliances}
        assert appliance_classnames.issubset(enduse_coords), (
            f"Appliances {appliance_classnames} not all in enduse coords {enduse_coords}"
        )
        # Hot-water flow must never exceed total flow at any point.
        total = consumption.sel(flowtypes='totalflow').values
        hot = consumption.sel(flowtypes='hotflow').values
        assert np.all(hot <= total + 1e-6), (
            "Hot flow exceeds total flow in at least one point."
        )