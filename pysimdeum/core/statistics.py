import os
import toml
from dataclasses import dataclass, field
from pysimdeum.utils.patterns import complex_daily_pattern
from pysimdeum.data import DATA_DIR


@dataclass
class Statistics:
    """Statistics dataclass that contains all the relevant statistical information for pysimdeum."""
    # Class attribute: same pointer for all instances.
    expected_end_uses = [
        'BathroomTap',
        'Bathtub',
        'Dishwasher',
        'KitchenTap',
        'OutsideTap',
        'Shower',
        'WashingMachine',
        'Wc',
    ]

    # Dataclass attribute (note the type declaration): unique for each instance.
    country: str = 'NL'
    household: dict = field(default_factory=dict)
    diurnal_pattern: dict = field(default_factory=dict)
    end_uses: dict = field(default_factory=dict)
    statisticsdir: str = ""  # TODO: Find good solution for this dirty statistics file workaround

    def __post_init__(self):
        # Check if pointing to a custom statistics directory or a country in the repository
        if os.path.isdir(self.country):
            self.statisticsdir = self.country
            self.country = None #No country is set as its a custom directory
        else:
            self.statisticsdir = os.path.join(DATA_DIR, self.country)

        # Load household statistics
        household_file = os.path.join(self.statisticsdir, 'household_statistics.toml')
        self.household = toml.load(open(household_file, 'r'))

        # Load diurnal pattern statistics
        diurnal_pattern_file = os.path.join(self.statisticsdir, 'diurnal_patterns.toml')
        self.diurnal_pattern = toml.load(open(diurnal_pattern_file, 'r'))

        # load end-uses:
        self.end_uses = dict()
        path2end_use = os.path.join(self.statisticsdir, 'end_uses')
        for end_use_name in self.expected_end_uses:
            with open(os.path.join(path2end_use, f'{end_use_name}.toml'), 'r') as end_use_file:
                self.end_uses[end_use_name] = self._convert_to_dict(toml.load(end_use_file))

            # Initialize special patterns
            if end_use_name == 'KitchenTap':
                self.end_uses[end_use_name]['daily_pattern'] = complex_daily_pattern(self.end_uses[end_use_name], freq='15Min')
            elif end_use_name in ['WashingMachine', 'Dishwasher']:
                self.end_uses[end_use_name]['daily_pattern'] = complex_daily_pattern(self.end_uses[end_use_name])

    def _convert_to_dict(self, data):
        """Convert dict subclasses to native Python `dict` for `pickle` to work.

        `toml.load()` returns `DynamicInlineTableDict`, a subclass of `dict`.
        It looks and behaves the same, until `pickle` tries to serialize it and fails.
        """
        if isinstance(data, dict):
            return {k: self._convert_to_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_dict(v) for v in data]
        else:
            return data