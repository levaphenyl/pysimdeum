import os
import pandas as pd
import numpy as np


def _subcatchment_profiles_to_dataframe(subcatchment_profiles):
    """
    Converts subcatchment_profiles (dict of xarray DataArrays) to a pandas DataFrame.

    Averages over the 'patterns' dimension to produce a single representative
    time series per zone. Returns a DataFrame indexed by datetime with one
    column per zone.

    Args:
        subcatchment_profiles (dict): Dict of {zone_id: xarray.DataArray} as
            returned by Population.subcatchment_profiles. Each DataArray is
            expected to have a 'time' dimension and optionally a 'patterns'
            dimension. Values are in L/s.

    Returns:
        pd.DataFrame: DataFrame indexed by datetime, columns = zone ids, values in L/s.
    """
    result = {}
    for zone_id, da in subcatchment_profiles.items():
        if 'patterns' in da.dims:
            series = da.mean('patterns').to_series()
        else:
            series = da.to_series()
        result[zone_id] = series
    df = pd.DataFrame(result)
    df.index = pd.to_datetime(df.index)
    return df


# Conversion factors from L/s to each supported Q_option unit
_LS_TO_Q = {
    'm3_h':   3.6,    # L/s * 3600 / 1000
    'm3_day': 86.4,   # L/s * 86400 / 1000
    'l_s':    1.0,
    'l_min':  60.0,
    'multipliers': None,  # handled separately
}


def generate_icm_csv(subcatchment_profiles, output_dir):
    """
    Generates a CSV file for each subcatchment wastewater profile formatted for InfoWorks ICM.

    One file per subcatchment is written to output_dir, named
    ``<subcatchment_id>_calibration.csv``.

    Args:
        subcatchment_profiles (dict): Dict produced by
            ``Population.subcatchment_ww_profiles``. Each value is a dict
            containing ``daily_flow``, ``hourly_average``, and ``ww_profile``.
        output_dir (str): Directory where the CSV files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for subcatchment_id, profile in subcatchment_profiles.items():
        daily_flow = profile['daily_flow']
        hourly_average = profile['hourly_average']
        ww_profile = profile['ww_profile']

        calibration_weekday = []
        for hour in range(24):
            hour_data = ww_profile[ww_profile['time'].dt.hour == hour]
            if not hour_data.empty:
                date = hour_data['time'].dt.date.iloc[0]
                flow_factor = hour_data['flow'].sum() / hourly_average[date]
                daily_flow_value = daily_flow[date]
            else:
                flow_factor = 0
                daily_flow_value = 0

            calibration_weekday.append({
                'TIME': f"{hour:02d}:00",
                'FLOW': round(flow_factor, 2),
                'POLLUTANT': 1
            })

        calibration_weekday_df = pd.DataFrame(calibration_weekday)
        calibration_weekend_df = calibration_weekday_df.copy()

        calibration_monthly = [
            {'MONTH': month, 'FLOW': 1, 'POLLUTANT': 1}
            for month in [
                "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
                "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"
            ]
        ]
        calibration_monthly_df = pd.DataFrame(calibration_monthly)

        design_profiles = [
            {'TIME': f"{hour:02d}:00", 'FLOW': 1, 'POLLUTANT': 1}
            for hour in range(24)
        ]
        design_profiles_df = pd.DataFrame(design_profiles)

        csv_content = [
            "!Version=1,type=WWG,encoding=UTF8",
            "TITLE,POLLUTANT_COUNT",
            "User defined WWG item,16",
            "Units_Concentration,Units_Salt_Concentration,Units_Temperature,Units_Average_Flow",
            "mg/l,kg/m3,degC,l/day",
            "PROFILE_NUMBER,PROFILE_DESCRIPTION,FLOW",
            f"1,1 Standard Profile {round(daily_flow_value)}l/day,{round(daily_flow_value)}",
            "SEDIMENT,AVERAGE_CONCENTRATION",
            "SF1,0",
            "SF2,0",
            "POLLUTANT,DISSOLVED,SF1,SF2",
            "BOD,0,0,0",
            "COD,0,0,0",
            "TKN,0,0,0",
            "NH4,0,0,0",
            "TPH,0,0,0",
            "PL1,0,0,0",
            "PL2,0,0,0",
            "PL3,0,0,0",
            "PL4,0,0,0",
            "DO_,0,0,0",
            "NO2,0,0,0",
            "NO3,0,0,0",
            "PH_,0,0,0",
            "SAL,0,0,0",
            "TW_,0,0,0",
            "COL,0,0,0",
            "CALIBRATION_WEEKDAY",
            "TIME,FLOW,POLLUTANT"
        ]
        csv_content += calibration_weekday_df.to_csv(index=False, header=False).splitlines()
        csv_content += ["CALIBRATION_WEEKEND", "TIME,FLOW,POLLUTANT"]
        csv_content += calibration_weekend_df.to_csv(index=False, header=False).splitlines()
        csv_content += ["CALIBRATION_MONTHLY", "MONTH,FLOW,POLLUTANT"]
        csv_content += calibration_monthly_df.to_csv(index=False, header=False).splitlines()
        csv_content += ["DESIGN_PROFILES", "TIME,FLOW,POLLUTANT"]
        csv_content += design_profiles_df.to_csv(index=False, header=False).splitlines()

        output_file = os.path.join(output_dir, f"{subcatchment_id}_calibration.csv")
        with open(output_file, 'w') as f:
            f.write("\n".join(csv_content))


def generate_ws_ddg(subcatchment_profiles, output_dir, Q_option='l_s', patternfile_option=1):
    """
    Writes InfoWorks WS demand diagram (.ddg) files from aggregated zone consumption profiles.

    Mirrors the behaviour of the MATLAB ``writeNonResPatternsToDdg`` function.
    The simulated profile is tiled to 7 days (the maximum InfoWorks supports).
    Two output artefacts are always written:

    * ``SPG_zone_pattern_file.ddg`` — all zones in one file (patternfile_option=1), or
      ``SPG_zone_pattern_file{n}.ddg`` for n=1..7 (patternfile_option=2)
    * ``SPG_zone_base_demand_file.txt`` — average flow per zone

    Args:
        subcatchment_profiles (dict): Dict of {zone_id: xarray.DataArray} —
            typically ``Population.subcatchment_profiles``. Values must be in L/s.
        output_dir (str): Directory where output files are written (created if absent).
        Q_option (str): Output flow unit. One of:
            ``'m3_h'``, ``'m3_day'``, ``'l_s'`` (default), ``'l_min'``,
            ``'multipliers'``. ``'multipliers'`` writes each timestep value as
            a factor relative to the zone mean (mean = 1).
        patternfile_option (int):
            ``1`` (default) — all zones in a single .ddg file.
            ``2`` — one .ddg file per day of the week (7 files).

    Raises:
        ValueError: If Q_option is not recognised or the inferred timestep is < 60 s.
    """
    valid_q = set(_LS_TO_Q.keys())
    if Q_option not in valid_q:
        raise ValueError(f"Q_option must be one of {sorted(valid_q)}, got '{Q_option}'")

    os.makedirs(output_dir, exist_ok=True)

    df = _subcatchment_profiles_to_dataframe(subcatchment_profiles)
    zone_ids = list(df.columns)

    timestep = int(pd.Timedelta(df.index[1] - df.index[0]).total_seconds())
    if timestep < 60:
        raise ValueError(
            f"Timestep inferred from data is {timestep}s — DDG format requires >= 60s. "
            "Resample your profiles before calling this function."
        )

    steps_per_day = 86400 // timestep
    max_num_days = 7  # InfoWorks only knows 7 days of the week
    total_steps_needed = max_num_days * steps_per_day

    # Read simulation duration directly from the time coordinate of the first zone
    first_da = next(iter(subcatchment_profiles.values()))
    sim_times = pd.to_datetime(first_da.coords['time'].values)
    num_simulated_days = (sim_times[-1] - sim_times[0]).days + 1
    # Use real data up to 7 days; if more than 7, crop; if less, wrap cyclically
    use_steps = min(num_simulated_days * steps_per_day, total_steps_needed)

    # Convert units then fill to exactly 7 days
    zone_values = {}
    for zone_id in zone_ids:
        vals = df[zone_id].values[:use_steps]  # L/s, up to 7 days of real data
        if Q_option == 'multipliers':
            mean_val = vals.mean()
            vals = vals / mean_val if mean_val != 0 else vals
        else:
            vals = vals * _LS_TO_Q[Q_option]
        # np.resize wraps cyclically — fills any remaining days from the start of vals
        zone_values[zone_id] = np.resize(vals, total_steps_needed)

    # DDG line builders

    _MAX_NAME = 9

    def _h1(pat_name, l_pat):
        trunc = pat_name[:_MAX_NAME]
        pad = max(0, 14 - len(trunc) - len(str(l_pat)))
        return f"{trunc}{' ' * pad}{l_pat}     0     0     0        {pat_name}"

    _h2 = "          " + "  ".join(["1.00"] * 12) + " 1 1"  # 12 monthly factors
    _h3 = "          " + "  ".join(["1.00"] * 7)              # 7 day-of-week factors

    def _data_row(step_in_day, value, day_num=None):
        total_secs = step_in_day * timestep
        h = total_secs // 3600
        m = (total_secs % 3600) // 60
        time_str = f"{h:02d}.{m:02d}"
        val_str = f"{value:.4f}" if value > 0 else "0"
        if day_num is not None:
            pad = max(0, 12 - len(val_str) - len(str(day_num)))
            row = f"         {time_str}{val_str}{' ' * pad}{day_num}"
        else:
            row = f"         {time_str}{val_str}"
        # from single-digit hours (00-09), e.g. ' 07.00' -> '  7.00'.
        return row.replace(' 0', '  ')

    _unit_comment = {
        'm3_h':       'zone demands are in m3/h',
        'm3_day':     'zone demands are in m3/day',
        'l_s':        'zone demands are in l/s',
        'l_min':      'zone demands are in l/min',
        'multipliers': 'zone demands are multiplier factors (average = 1)',
    }

    if patternfile_option == 1:
        out_path = os.path.join(output_dir, 'SPG_zone_pattern_file.ddg')
        with open(out_path, 'w') as f:
            f.write('SIMDEUM Pattern Generator - patterns\n')
            for zone_id in zone_ids:
                vals = zone_values[zone_id]
                pat_name = f"S_{zone_id}"
                f.write(_h1(pat_name, len(vals)) + '\n')
                f.write(_h2 + '\n')
                f.write(_h3 + '\n')
                for i, val in enumerate(vals):
                    day_num = i // steps_per_day + 1
                    f.write(_data_row(i % steps_per_day, val, day_num=day_num) + '\n')
            f.write('END\n')

    else:  # one file per day of the week
        for day in range(max_num_days):
            out_path = os.path.join(output_dir, f'SPG_zone_pattern_file{day + 1}.ddg')
            with open(out_path, 'w') as f:
                f.write('SIMDEUM Pattern Generator - patterns\n')
                for zone_id in zone_ids:
                    day_vals = zone_values[zone_id][day * steps_per_day:(day + 1) * steps_per_day]
                    pat_name = f"S_{zone_id}"
                    f.write(_h1(pat_name, len(day_vals)) + '\n')
                    f.write(_h2 + '\n')
                    f.write(_h3 + '\n')
                    for i, val in enumerate(day_vals):
                        f.write(_data_row(i, val) + '\n')
                f.write('END\n')

    base_path = os.path.join(output_dir, 'SPG_zone_base_demand_file.txt')
    with open(base_path, 'w') as f:
        f.write('; SIMDEUM Pattern Generator - average zone demand\n')
        f.write(f'; {_unit_comment[Q_option]}\n')
        f.write('; \n\n')
        f.write(' [BASE DEMANDS]\n')
        f.write('; ID demands\n')
        for zone_id in zone_ids:
            avg = float(np.mean(zone_values[zone_id]))
            f.write(f'SPG_{zone_id} {avg:.4f}\n')
