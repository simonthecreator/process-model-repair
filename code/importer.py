import pm4py
import pandas as pd
import pathlib

def read_from_xes(filename):
    return pm4py.read_xes(filename)

def read_durations(file_path: str):
    file_path = file_path.split(".")[0] + '_durations.csv'
    return pd.read_csv(file_path)

dateparser = lambda x: pd.to_datetime(x, format="%d.%m.%Y %H:%M:%S")

def read_from_input_file(file_path: pathlib.Path):
    file_path = file_path.__str__()

    if file_path.endswith(".csv"):
        d_types = {'case:concept:name': str, 'concept:name': str}
        data = pd.read_csv(file_path,
                        dtype=d_types,
                        date_parser=dateparser,
                        parse_dates=['time:timestamp'],
                        header=0)

    if file_path.endswith(".xes"):
        data = pm4py.read_xes(file_path)

    return data