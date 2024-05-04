import pandas as pd
import numpy as np

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def remove_outliers_in_kpi_values(df: pd.DataFrame, outliers_threshold: int) -> pd.DataFrame:
    """OEE-Aspect `Performance` contains outliers, values that are far higher than 1, even if Performance is defined as the ratio of Processing duration divided by Planned processing duration.
    It is supposed to be around 1, but if there are faulty or wrong processing durations tracked or errors in planning, there are outlier values that are much higher.

    Outlier Remove: set all `Performance` values that are > outliers_threshold to 1.
    We do not want to remove outliers since each process step is part of a trace and for model repair we either incorporate a whole trace to the model or not at all.
    Therefore, each process step can be a valuable part of a trace even if Performance of this very step is an outlier and is set to 1 here.

    Args:
        df (pd.DataFrame): df with a column named `Performance`
        outliers_threshold (int): values above this threshold are set to 1

    Returns:
        pd.DataFrame: same df as input but removed outliers in column `Performance`
    """
    performance_outliers = np.array(df['Performance'].values.tolist())
    df['Performance'] = np.where(performance_outliers > outliers_threshold, 1, performance_outliers).tolist()
    return df