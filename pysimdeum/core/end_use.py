import copy
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pysimdeum.utils.probability import (
    chooser,
    normalize,
    to_timedelta,
    sample_value,
)
from pysimdeum.utils.patterns import handle_spillover_consumption, handle_discharge_spillover, sample_start_time, offset_simultaneous_discharge


#TODO: Specific EndUse __post_init__ calls can be replaced by directly using the class name instead of setting the name attributes

@dataclass
class EndUse:
    """Base class for end-uses."""
    # Class attributes (shared across all instances).
    cold_water_temp = 10
    hot_water_temp = 60

    # Instance attributes.
    statistics: dict = field(repr=False)  # dict object from core.Statistics.end_uses associated with the end-use
    name: str = "EndUse"  # ... name of the end-use
    discharge_events: list = field(default_factory=list)
    intensity: float = 0.  # Flow rate for each end use in L/s.

    def __post_init__(self):
        """Initialize field values that depend on other fields, among others."""
        self.offset = int(pd.Timedelta(self.statistics['offset']).total_seconds())
        if 'intensity' in self.statistics:
            if isinstance(self.statistics['intensity'], dict):
                distribution_name, distribution_params = self.get_statistical_params(self.statistics['intensity'])
                self.intensity = sample_value(distribution_name, **distribution_params)
            else:
                self.intensity = self.statistics['intensity']

    def init_consumption(self, users: list, time_resolution: str='1s') -> pd.DataFrame:
        """Initialization of a pandas dataframe to store the  consumptions.

        Args:
            users:  list with users
            time_resolution:  string with desired time resolution as python pandas `DateOffset` object
            (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects)

        Returns:
            consumption as pandas `DataFrame` filled with zeros

        Raise:
            KeyError:   If no users are provided.
        """

        if users:
            # produce datetime index
            index = pd.timedelta_range(start='00:00:00', end='24:00:00', freq=time_resolution, closed='left')

            # name columns by users
            columns = ['user_' + str(x+1) for x, user in enumerate(users)]

            # initialise consumption dataframe with timedelta index and user columnnames, name it according to end-use
            # device and fill it with zeros.
            consumption = pd.DataFrame(data=0, index=index, columns=columns)
        else:
            # raise an error if no users are defined.
            raise KeyError('No Users are defined!')

        return consumption

    @staticmethod
    def usage_probability(time_resolution: str='1s') -> pd.Series:
        """Produces uninformed prior.

        For more specific usage probabilities (washing machine, kitchen tap, dishwasher) overload this function by
        loading a usage pattern into it.
        """
        # produce datetime index
        index = pd.timedelta_range(start='00:00:00', end='24:00:00', freq=time_resolution, closed='left')

        prob = pd.Series(data=1, index=index)  # ... uniform probability over time and cast it into pandas series.
        prob /= prob.sum()  # ... normalization of the probabilities

        return prob

    def get_statistical_params(self, dist_config: dict, age: str|None=None, gender: str|None=None, numusers: int|None=None) -> tuple[str, dict]:
        """Split the statistical configuration into a distribution name and distribution parameters.

        Use before calling `utils.probability.sample_value`.
        This function supports the following use cases:
            - The parameters of the statistical distribution are constant.
            - The parameters of the statistical distribution depend on the size of the household.
            - The parameters of the statistical distribution depend on the age category of the user.
            - The parameters of the statistical distribution depend on the age category of the user and their gender.

        Args:
            dist_config:    The configuration of the statistical distribution, as found in `Statistics.end_uses`.
                            It **must** have a key called `'distribution'`.
            age:            The age category of the user (child, teen, work_ad, home_ad, senior, total).
                            Use if the distribution parameters depend on it.
            gender:         The gender of the user (female, male).
                            Use if the distribution parameters depend on it.
            numusers:       The number of users, i.e. the household size.
                            Use if the distribution parameters depend on it.

        Returns:
            dist_name:      The name of the distribution.
            dist_params:    The parameters of the distribution, as a single-level map.

        Raise:
            KeyError:       If `dist_config` has no `'distribution'` key, or if the given `age`, `gender`, or `numusers` is not found in the nested parameter dict.
        """
        dist_config_copy = copy.deepcopy(dist_config)  # Avoid any side-effects of the function.
        dist_name = dist_config_copy.pop('distribution')
        if any([age, gender, numusers]):
            dist_params = {}
            for param_name, param_value in dist_config_copy.items():
                if isinstance(param_value, dict):
                    if age is not None and gender is not None:
                        dist_params[param_name] = param_value[age][gender]
                    elif age is not None:
                        dist_params[param_name] = param_value[age]
                    elif numusers is not None:
                        dist_params[param_name] = param_value[str(numusers)]
                else:
                    dist_params[param_name] = param_value
        else:
            dist_params = dist_config_copy

        return dist_name, dist_params

    def fct_frequency(self) -> int:
        """Samples a number of events from the frequency probability function defined in the EndUse configuration.

        Simple case where the distribution parameters are constant.

        Returns:
            An integer value representing the number of events for the given day.
        """
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'])
        return round(sample_value(dist_name, **dist_params))

    def fct_duration(self):
        """Placeholder for specific duration probability function defined in specific EndUse"""

        raise NotImplementedError('Duration function is not implemented yet!')

    def fct_intensity(self):
        """Placeholder for specific intensity probability function defined in specific EndUse"""

        raise NotImplementedError('Intensity function is not implemented yet!')

    def temperature(self):
        """Placeholder for specific temperature function defined in specific EndUse"""

        raise NotImplementedError('temperature function is not implemented yet!')

    def fct_duration_intensity_temperature(self) -> tuple[int, float, float]:
        """Computes event duration and flow rate intensity and temperature for the end use.

        Use this function if the duration, intensity, and temperature depend on each other.
        For instance, they all depend on the subtype (washing hands, etc.).
        Side effect: add the `subtype` attribute to the end use object.

        Returns:
            duration:       The total time of the water use, in seconds.
            intensity:      The flow rate of the water use, in liters per second.
            temperature:    The water temperature, in degrees Celsius.
        """
        self.subtype = chooser(self.statistics['subtype'], 'penetration')
        d_dist, d_stats = self.get_statistical_params(self.statistics['subtype'][self.subtype]['duration'])
        duration = round(sample_value(d_dist, **d_stats))
        i_dist, i_stats = self.get_statistical_params(self.statistics['subtype'][self.subtype]['intensity'])
        intensity = sample_value(i_dist, **i_stats)
        if self.intensity > 0.:
            intensity *= self.intensity  # Expect subtype intensity to be a fraction of the end-use intensity if defined.

        temperature = self.statistics['subtype'][self.subtype]['temperature']
        return duration, intensity, temperature


@dataclass
class Bathtub(EndUse):
    """Class for Bathtub end-use."""
    name: str = "Bathtub"
    wastewater_type: str = "greywater"

    def fct_frequency(self, age:str|None=None):
        """Random function computing the frequency of use for the Bathtub end-use class.

        Args:
            age: age category of the user of the user (child, teen, etc.).

        Returns:
            A sampled value of the frequency of use from the distribution.
        """
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'], age=age)
        return round(sample_value(dist_name, **dist_params))

    def fct_duration(self):
        """Compute the duration of Bathtub end-use.

        If volume is specified in statistics, calculates duration from volume and intensity.
        Otherwise, uses a fixed duration from statistics.

        Returns:
            duration (integer in seconds)
        """
        # Use volume if available.
        if 'volume' in self.statistics:
            if isinstance(self.statistics['volume'], dict):
                dist_name, dist_params = self.get_statistical_params(self.statistics['volume'])
                volume = sample_value(dist_name, **dist_params)
            else:
                volume = self.statistics['volume']

            return int(volume / self.intensity)  # L / L/s = s

        # fixed duration
        return int(to_timedelta(self.statistics['duration']).total_seconds())

    def fct_intensity(self):
        """Compute the intensity of Bathtub end-use.

        Returns:
            intensity (constant or sampled float value as configured)
        """
        return self.intensity

    def temperature(self):
        """Obtain the temperature of a bath

        Returns:
            temperature of bath water

        """
        # independent of subtype
        return self.statistics['temperature']

    def calculate_discharge(self, discharge, end, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num):
        remaining_water = intensity * duration
        # Sample a usage_delay from the distribution
        dist_name, dist_params = self.get_statistical_params(self.statistics['usage_delay'])
        usage_delay = sample_value(dist_name, **dist_params) * 60.
        start = int(end + usage_delay)

        # Sample a value from the discharge_intensity distribution
        dist_name, dist_params = self.get_statistical_params(self.statistics['discharge_intensity'])
        discharge_flow_rate = sample_value(dist_name, **dist_params)

        self.discharge_events.append({
            'enduse': self.name,
            'usage': self.name, # no bath subtypes
            'start': start,
            'end': int(start + (remaining_water / discharge_flow_rate)),
            'discharge_temperature': self.statistics['discharge_temperature'],
        })

        while remaining_water > 0:
            discharge_duration = remaining_water / discharge_flow_rate
            end = int(start + discharge_duration)
            discharge[start:end, j, ind_enduse, pattern_num, 0] = discharge_flow_rate
            remaining_water -= discharge_flow_rate * discharge_duration
            start = end

        return discharge

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = self.usage_probability().values

        previous_events = []

        for j, user in enumerate(users):
            freq = self.fct_frequency(age=user.age)
            prob_user = user.presence.values

            for i in range(freq):
                duration = self.fct_duration()
                intensity = self.fct_intensity()
                temperature = self.temperature()
                prob_joint = normalize(prob_user * prob_usage)

                start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
                previous_events.append((start, end))

                consumption[start:end, j, ind_enduse, pattern_num, 0] = intensity
                temperature_fraction = (temperature - self.cold_water_temp)/(self.hot_water_temp - self.cold_water_temp)
                consumption[start:end, j, ind_enduse, pattern_num, 1] = intensity*temperature_fraction

                if simulate_discharge:
                    if discharge is None:
                        raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                    discharge = self.calculate_discharge(discharge, end, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class BathroomTap(EndUse):
    """Base class for bathroom taps."""
    name: str = "BathroomTap"
    wastewater_type: str = "greywater"

    def calculate_discharge(self, discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, spillover=False):
        remaining_water = intensity * duration
        start = int(start)

        # Sample a value from the discharge_intensity distribution
        dist_name, dist_params = self.get_statistical_params(self.statistics['subtype'][self.subtype]['discharge_intensity'])
        discharge_flow_rate = sample_value(dist_name, **dist_params)

        # limit discharge_flow_rate to the intensity of the tap if there is not enough water to discharge
        if discharge_flow_rate > intensity:
            discharge_flow_rate = intensity

        start = offset_simultaneous_discharge(discharge, start, j, ind_enduse, pattern_num, spillover=spillover)
        self.discharge_events.append({
            'enduse': self.name,
            'usage': self.subtype, # subtypes are inherited from chooser(toml)
            'start': start,
            'end': int(start + (remaining_water / discharge_flow_rate)),
            'discharge_temperature': self.statistics['subtype'][self.subtype]['discharge_temperature'],
        })

        while remaining_water > 0:
            discharge_duration = remaining_water / discharge_flow_rate
            end = int(start + discharge_duration)            
            discharge[start:end, j, ind_enduse, pattern_num, 0] = discharge_flow_rate
            remaining_water -= discharge_flow_rate * discharge_duration
            start = end

        return discharge

    def simulate(self, consumption, discharge, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):
        prob_usage = self.usage_probability().values
        previous_events = []

        for j, user in enumerate(users):
            freq = self.fct_frequency()
            prob_user = user.presence.values

            for i in range(freq):

                duration, intensity, temperature = self.fct_duration_intensity_temperature()

                prob_joint = normalize(prob_user * prob_usage)

                start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
                previous_events.append((start, end))

                consumption[start:end, j, ind_enduse, pattern_num, 0] = intensity
                temperature_fraction = (temperature - self.cold_water_temp)/(self.hot_water_temp - self.cold_water_temp)
                consumption[start:end, j, ind_enduse, pattern_num, 1] = intensity*temperature_fraction

                if simulate_discharge:
                    if discharge is None:
                        raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                    discharge = self.calculate_discharge(discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, spillover=spillover)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class Dishwasher(EndUse):
    """Base class for dishwashers."""
    name: str = "Dishwasher"
    wastewater_type: str = "blackwater"

    def fct_frequency(self, numusers=None):
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'], numusers=numusers)
        return round(sample_value(dist_name, **dist_params))

    def fct_duration_pattern(self, start=None):
        pattern = self.statistics['enduse_pattern']
        return pattern

    def calculate_discharge(self, discharge, start, j, ind_enduse, pattern_num, day_num, end_of_day, total_days, spillover=False):
        discharge_pattern = self.statistics['discharge_pattern']
        cycle_times = []

        for time in discharge_pattern[discharge_pattern > 0].index:
            discharge_time  = start + int(time.total_seconds())
            if discharge_time > end_of_day and spillover:
                discharge = handle_discharge_spillover(discharge, discharge_pattern, time, discharge_time, j, ind_enduse, pattern_num, end_of_day, total_days)
            elif ((day_num + 1) == total_days) and (discharge_time > end_of_day):
                pass
            else:
                discharge[discharge_time, j, ind_enduse, pattern_num, 1] = discharge_pattern[time]

                if not cycle_times or discharge_time - cycle_times[-1][1] > 1:
                    cycle_times.append([discharge_time, discharge_time])
                else:
                    cycle_times[-1][1] = discharge_time

        discharge_temperature = self.statistics['discharge_temperature']

        if isinstance(discharge_temperature, (int, float)):
            discharge_temperatures = [discharge_temperature] * len(cycle_times)
        elif isinstance(discharge_temperature, dict):
            dist_name, dist_params = self.get_statistical_params(discharge_temperature)
            discharge_temperatures = [sample_value(dist_name, **dist_params) for __ in cycle_times]
        else:
            raise ValueError("Discharge temperature type not implemented.")

        self.discharge_events.append({
            'enduse': self.name,
            'usage': self.name, # no subtypes currently
            'start': [cycle[0] for cycle in cycle_times],
            'end': [cycle[1] for cycle in cycle_times],
            'discharge_temperature': discharge_temperatures,
        })

        return discharge

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = copy.deepcopy(self.statistics['daily_pattern'].values)
        freq = self.fct_frequency(numusers=len(users))

        for j, user in enumerate(users):
            if j == 0:
                prob_user = copy.deepcopy(user.presence)
            else:
                prob_user += user.presence

        prob_user = normalize(prob_user.values)
        j = len(users)

        prob_joint = normalize(prob_user * prob_usage)

        pattern = self.fct_duration_pattern().values
        duration = len(pattern)

        previous_events = []

        for i in range(freq):
            start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)

            # add event times to list of previous events
            previous_events.append((start, end))

            end_of_day = 24 * 60 * 60 * (day_num + 1)
            if end > end_of_day and spillover:
                consumption = handle_spillover_consumption(consumption, pattern, start, end, j, ind_enduse, pattern_num, end_of_day, self.name, total_days)
            elif ((day_num + 1) == total_days) and (end > end_of_day):
                difference = end_of_day - start
                consumption[start:end_of_day, j, ind_enduse, pattern_num, 0] = pattern[:difference]
                consumption[start:end_of_day, j, ind_enduse, pattern_num, 1] = 0
            else:
                difference = end - start
                consumption[start:end, j, ind_enduse, pattern_num, 0] = pattern[:difference]
                consumption[start:end, j, ind_enduse, pattern_num, 1] = 0

            if simulate_discharge:
                if discharge is None:
                    raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                discharge = self.calculate_discharge(discharge, start, j, ind_enduse, pattern_num, day_num, end_of_day, total_days, spillover=spillover)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class KitchenTap(EndUse):
    """Base class for kitchen taps."""
    name: str = "KitchenTap"
    wastewater_type: str = "blackwater"

    def fct_frequency(self, numusers=None):
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'], numusers=numusers)
        return round(sample_value(dist_name, **dist_params))

    def calculate_discharge(self, discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, usage, spillover=False):
        remaining_water = intensity * duration
        start = int(start)

        # Sample a value from the discharge_intensity distribution
        dist_name, dist_params = self.get_statistical_params(self.statistics['subtype'][self.subtype]['discharge_intensity'])
        discharge_flow_rate = sample_value(dist_name, **dist_params)

        # limit discharge_flow_rate to the intensity of the tap if there is not enough water to discharge
        if discharge_flow_rate > intensity:
            discharge_flow_rate = intensity

        # Check if the tap is turned off before the end of the duration, if so, update the start time
        start = offset_simultaneous_discharge(discharge, start, j, ind_enduse, pattern_num, spillover=spillover)

        if discharge_flow_rate > 0.:
            self.discharge_events.append({
                'enduse': self.name,
                'usage': usage, # subtypes are from chooser(toml)
                'start': start,
                'end': int(start + (remaining_water / discharge_flow_rate)),
                'discharge_temperature': self.statistics['subtype'][self.subtype]['discharge_temperature'],
            })

            while remaining_water > 0:
                discharge_duration = remaining_water / discharge_flow_rate
                end = int(start + discharge_duration)
                # check if subtype = consumption (drinking), if so the discharge flow rate is set to 0
                if self.subtype == 'consumption':
                    discharge[start:end, j, ind_enduse, pattern_num, 1] = 0
                else:
                    discharge[start:end, j, ind_enduse, pattern_num, 1] = discharge_flow_rate
                remaining_water -= discharge_flow_rate * discharge_duration
                start = end

        # else: the event is consumption and there is no discharge.

        return discharge

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = copy.deepcopy(self.statistics['daily_pattern'].values)

        for j, user in enumerate(users):
            if j == 0:  # ToDo: Add function that computes usage probability for all users
                prob_user = copy.deepcopy(user.presence)
            else:
                prob_user += user.presence

        prob_user = normalize(prob_user).values

        j = len(users)

        freq = self.fct_frequency(numusers=len(users))

        previous_events = []

        for i in range(freq):

            duration, intensity, temperature = self.fct_duration_intensity_temperature()

            # assign usage type (based on subtype)
            usage = self.subtype
            prob_joint = normalize(prob_user * prob_usage)  # ToDo: Check if joint probability can be computed outside of for loop for all functions
            start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
            previous_events.append((start, end))

            consumption[start:end, j, ind_enduse, pattern_num, 0] = intensity
            temperature_fraction = (temperature - self.cold_water_temp)/(self.hot_water_temp - self.cold_water_temp)
            consumption[start:end, j, ind_enduse, pattern_num, 1] = intensity*temperature_fraction

            if simulate_discharge:
                if discharge is None:
                    raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                discharge = self.calculate_discharge(discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, usage, spillover=spillover)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class OutsideTap(EndUse):
    """Base class for outdoor water use."""
    name: str = "OutsideTap"

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = self.usage_probability().values

        freq = 0
        for j, user in enumerate(users):
            if j == 0:
                prob_user = copy.deepcopy(user.presence)
            else:
                prob_user += user.presence
            freq += self.fct_frequency()

        prob_user = normalize(prob_user).values

        j = len(users)

        previous_events = []

        for i in range(freq):

            duration, intensity, temperature = self.fct_duration_intensity_temperature()

            prob_joint = normalize(prob_user * prob_usage)
            start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
            previous_events.append((start, end))

            consumption[start:end, j, ind_enduse, pattern_num, 0] = intensity
            temperature_fraction = (temperature - self.cold_water_temp)/(self.hot_water_temp - self.cold_water_temp)
            consumption[start:end, j, ind_enduse, pattern_num, 1] = intensity*temperature_fraction

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class Shower(EndUse):
    """Base class for all showers."""
    name: str = "Shower"
    wastewater_type: str = "greywater"

    def fct_frequency(self, age=None):
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'], age=age)
        return round(sample_value(dist_name, **dist_params))

    def fct_duration_intensity_temperature(self, age=None):
        d_dist, d_stats = self.get_statistical_params(self.statistics['duration'], age=age)
        duration = round(sample_value(d_dist, **d_stats))
        intensity = self.statistics['subtype'][self.name]['intensity']
        temperature = self.statistics['temperature']
        return duration, intensity, temperature

    def calculate_discharge(self, discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, spillover=False):
        remaining_water = intensity * duration

        start = int(start)

        # Sample a value from the discharge_intensity distribution
        discharge_flow_rate = self.statistics['subtype'][self.name]['discharge_intensity']

        # limit discharge_flow_rate to the intensity of the tap if there is not enough water to discharge
        if discharge_flow_rate > intensity:
            discharge_flow_rate = intensity

        start = offset_simultaneous_discharge(discharge, start, j, ind_enduse, pattern_num, spillover=spillover)

        self.discharge_events.append({
            'enduse': "Shower",
            'usage': "Shower", # subtypes are class inheritance names
            'start': start,
            'end': int(start + (remaining_water / discharge_flow_rate)),
            'discharge_temperature': self.statistics['discharge_temperature'],
        })

        while remaining_water > 0:
            discharge_duration = remaining_water / discharge_flow_rate
            end = int(start + discharge_duration)
            discharge[start:end, j, ind_enduse, pattern_num, 0] = discharge_flow_rate
            remaining_water -= discharge_flow_rate * discharge_duration
            start = end

        return discharge

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = self.usage_probability().values
        previous_events = []

        for j, user in enumerate(users):
            freq = self.fct_frequency(age=user.age)
            prob_user = user.presence.values

            for i in range(freq):
                duration, intensity, temperature = self.fct_duration_intensity_temperature(age=user.age)

                prob_joint = normalize(prob_user * prob_usage)
                start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
                previous_events.append((start, end))

                consumption[start:end, j, ind_enduse, pattern_num, 0] = intensity
                temperature_fraction = (temperature - self.cold_water_temp)/(self.hot_water_temp - self.cold_water_temp)
                consumption[start:end, j, ind_enduse, pattern_num, 1] = intensity*temperature_fraction

                if simulate_discharge:
                    if discharge is None:
                        raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                    discharge = self.calculate_discharge(discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, spillover=spillover)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class NormalShower(Shower):
    """Most common shower."""
    name: str = "NormalShower"


@dataclass
class FancyShower(Shower):
    """Combi-heater with water-saving showerhead."""
    name: str = "FancyShower"


@dataclass
class WashingMachine(EndUse):
    """Base class for washing machines."""
    name: str = "WashingMachine"
    wastewater_type: str = "blackwater"

    def fct_frequency(self, numusers=None):
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'], numusers=numusers)
        return round(sample_value(dist_name, **dist_params))

    def fct_duration_pattern(self, start=None):
        pattern = self.statistics['enduse_pattern']
        # duration = pattern.index[-1] - pattern.index[0]
        return pattern

    def calculate_discharge(self, discharge, start, j, ind_enduse, pattern_num, day_num, end_of_day, total_days, spillover=False):
        discharge_pattern = self.statistics['discharge_pattern']

        cycle_times = []

        for time in discharge_pattern[discharge_pattern > 0].index:
            discharge_time  = start + int(time.total_seconds())
            if discharge_time > end_of_day and spillover:
                discharge = handle_discharge_spillover(discharge, discharge_pattern, time, discharge_time, j, ind_enduse, pattern_num, end_of_day, total_days)
            elif ((day_num + 1) == total_days) and (discharge_time > end_of_day):
                pass
            else:
                discharge[discharge_time, j, ind_enduse, pattern_num, 1] = discharge_pattern[time]

            if not cycle_times or discharge_time - cycle_times[-1][1] > 1:
                    cycle_times.append([discharge_time, discharge_time])
            else:
                    cycle_times[-1][1] = discharge_time

        discharge_temperature = self.statistics['discharge_temperature']

        if isinstance(discharge_temperature, (int, float)):
            discharge_temperatures = [discharge_temperature] * len(cycle_times)
        elif isinstance(discharge_temperature, dict):
            dist_name, dist_params = self.get_statistical_params(discharge_temperature)
            discharge_temperatures = [sample_value(dist_name, **dist_params) for __ in cycle_times]
        else:
            raise ValueError("Discharge temperature type not implemented.")

        self.discharge_events.append({
            'enduse': "WashingMachine",
            'usage': "WashingMachine", # no subtypes currently
            'start': [cycle[0] for cycle in cycle_times],
            'end': [cycle[1] for cycle in cycle_times],
            'discharge_temperature': discharge_temperatures,
        })

        return discharge

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = copy.deepcopy(self.statistics['daily_pattern'].values)

        # for j, user in enumerate(users):
        freq = self.fct_frequency(numusers=len(users))

        for j, user in enumerate(users):
            if j == 0:
                prob_user = copy.deepcopy(user.presence)
            else:
                prob_user += user.presence

        prob_user = normalize(prob_user).values
        j = len(users)

        prob_joint = normalize(prob_user * prob_usage)

        pattern = self.fct_duration_pattern()
        duration = len(pattern)

        previous_events = []

        for i in range(freq):
            start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
            # add event times to list of previous events
            previous_events.append((start, end))

            end_of_day = 24 * 60 * 60 * (day_num + 1)
            if end > end_of_day and spillover:
                consumption = handle_spillover_consumption(consumption, pattern, start, end, j, ind_enduse, pattern_num, end_of_day, "WashingMachine", total_days)
            elif ((day_num + 1) == total_days) and (end > end_of_day):
                difference = end_of_day - start
                consumption[start:end_of_day, j, ind_enduse, pattern_num, 0] = pattern[:difference]
                consumption[start:end_of_day, j, ind_enduse, pattern_num, 1] = 0
            else:
                difference = end - start
                consumption[start:end, j, ind_enduse, pattern_num, 0] = pattern[:difference]
                consumption[start:end, j, ind_enduse, pattern_num, 1] = 0

            if simulate_discharge:
                if discharge is None:
                    raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                discharge = self.calculate_discharge(discharge, start, j, ind_enduse, pattern_num, day_num, end_of_day, total_days, spillover=spillover)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class Wc(EndUse):
    """Base class for all WC flushes."""
    name: str = "Wc"
    wastewater_type: str = "blackwater"

    def __post_init__(self):
        super().__post_init__()
        self.discharge_events = []

    def fct_frequency(self, age=None, gender=None):
        dist_name, dist_params = self.get_statistical_params(self.statistics['frequency'], age=age, gender=gender)
        return round(sample_value(dist_name, **dist_params))

    def fct_duration_intensity_temperature(self):

        flush_interuption = self.statistics['subtype'][self.name]['flush_interuption']
        prob_flush_interuption = self.statistics['prob_flush_interuption']

        temperature = self.statistics['temperature']
        average = to_timedelta(self.statistics['subtype'][self.name]['duration'])

        # add water savings option
        if flush_interuption:
            v = np.random.random() * 100
            if v < prob_flush_interuption:
                average /= 2.0

        duration = int(average.total_seconds())

        return duration, self.intensity, temperature

    def calculate_discharge(self, discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, usage):
        incoming_water = intensity * duration
        end = int(start)

        # Sample a value from the discharge_intensity distribution
        discharge_flow_rate = self.statistics['discharge_intensity']

        self.discharge_events.append({
            'enduse': "Wc",
            'usage': usage,
            'start': int(end - (incoming_water / discharge_flow_rate)),
            'end': end,
            'discharge_temperature': self.statistics['discharge_temperature'],
        })

        while incoming_water > 0:
            discharge_duration = incoming_water / discharge_flow_rate
            start = int(end - discharge_duration)
            discharge[start:end, j, ind_enduse, pattern_num, 1] = discharge_flow_rate
            incoming_water -= discharge_flow_rate * discharge_duration
            end = start

        return discharge

    def simulate(self, consumption, discharge=None, users=None, ind_enduse=None, pattern_num=1, day_num=0, total_days=1, simulate_discharge=False, spillover=False):

        prob_usage = self.usage_probability().values

        previous_events = []

        for j, user in enumerate(users):
            freq = self.fct_frequency(age=user.age, gender=user.gender)
            prob_user = user.presence.values

            for i in range(freq):

                duration, intensity, temperature = self.fct_duration_intensity_temperature()

                # assign usage type (urine or faeces)
                usage = "urine" if np.random.random() * 100 < self.statistics['prob_urine'] else "faeces"
                #print(prob_user)
                prob_joint = normalize(prob_user * prob_usage)
                start, end = sample_start_time(prob_joint, day_num, duration, previous_events, self.offset)
                previous_events.append((start, end))

                consumption[start:end, j, ind_enduse, pattern_num, 0] = intensity
                temperature_fraction = (temperature - self.cold_water_temp)/(self.hot_water_temp - self.cold_water_temp)
                consumption[start:end, j, ind_enduse, pattern_num, 1] = intensity*temperature_fraction

                if simulate_discharge:
                    if discharge is None:
                        raise ValueError("Discharge array is None. It must be initialized before being passed to the simulate function.")
                    discharge = self.calculate_discharge(discharge, start, duration, intensity, temperature_fraction, j, ind_enduse, pattern_num, usage)

        return consumption, (discharge if simulate_discharge else None)


@dataclass
class WcNormal(Wc):
    """Most common toilet flush."""
    name: str = 'WcNormal'


@dataclass
class WcNormalSave(Wc):
    """Most common toilet flush with a water-saving option."""
    name: str = "WcNormalSave"


@dataclass
class WcNew(Wc):
    """Toilet flush complying with the new efficiency standards."""
    name: str = "WcNew"


@dataclass
class WcNewSave(Wc):
    """Toilet flush complying with the new standard and with a water-saving option."""
    name: str = "WcNewSave"
