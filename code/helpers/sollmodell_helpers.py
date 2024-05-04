"""
Funktionen um ein Soll-Modell aus einem df zu erstellen 
"""
import pm4py
import pandas as pd
import datetime

def add_meldenr_as_timestamp(df):
    """Timestamp (Start- und Endzeitpunkt) aus Meldenummer erstellen (in Sekunden) und in XES-Standard column namen umwandeln"""
    for ts_col_name, meldenr_col_name in zip(['time:timestamp', 'start_timestamp'], ['Maximum Meldenummer', 'Minimum Meldenummer']):
        df[ts_col_name] = df.apply(
        lambda x: datetime.datetime.fromtimestamp(x[meldenr_col_name]).strftime("%Y-%m-%d %H:%M:%S"), axis=1)
        df[ts_col_name] = pd.to_datetime(df[ts_col_name])
    return df

def nacharbeit_herausfiltern(df: pd.DataFrame, nacharbeit_relnr: list = []) -> pd.DataFrame:
    """
    Nacharbeit herausfiltern, i.e. Arbeitsgänge, deren RelNr mit einem String beginnt, der in der Liste nacharbeit_relnr ist, werden entfernt
    """
    nacharbeit_relnr = [relnr for relnr in df['concept:name'].unique() if relnr.startswith(tuple(nacharbeit_relnr))]
    ohne_na = df[~df['concept:name'].isin(nacharbeit_relnr)]
    return ohne_na


def create_soll_agrelnr(df: pd.DataFrame):
    """Erstelle soll-log (Log für ein soll-Modell) aus df

    Args:
        df (DataFrame): df für das ein soll-log erstellt werden soll

    Returns:
        DataFrame: fiktiver Soll-Log mit den Events für ein Soll-Modell. Besteht nur aus einem Trace, der jedoch doppelt vorkommt für eindeutigen dfg
    """
    # Nacharbeit herausfiltern
    ohne_na = nacharbeit_herausfiltern(df, ['09'])

    # df erstellen und AG-RelNr sortieren
    soll_df = pd.DataFrame({'RelNr_str': ohne_na['concept:name'].unique()})
    soll_df = soll_df.merge(ohne_na['concept:name'], left_on='RelNr_str', right_on='concept:name', how='left')
    soll_df['RelNr_int'] = soll_df['RelNr_str'].astype(int)
    soll_df = soll_df.sort_values('RelNr_int').reset_index(drop=True)
    # deduplizieren mit Annahme dass Arbeitsgänge die gleiche RelNr aber leicht unterschiedliche Bezeichnung haben, dennoch gleich sind
    soll_df = soll_df.drop_duplicates(subset=['RelNr_int'])
    # komma aus concept:name entfernen, da in parse_event_log_string() als separator verwendet
    soll_df['concept:name'] = soll_df['concept:name'].str.replace(',', '')
    # soll-log erstellen (2x gleicher Trace für eindeutigen dfg)
    soll_log = pm4py.utils.parse_event_log_string([','.join(soll_df['concept:name'].to_list())]*2)
    soll_log['time:timestamp'] = pd.to_datetime(soll_log['time:timestamp'])#, format="%d.%m.%Y %H:%M:%S")
    soll_log['time:timestamp'] = soll_log['time:timestamp'].astype('datetime64[ns, UTC]')
    return soll_log

def create_soll_modell_by_variants(log: pd.DataFrame,
                                   activity_key: str = 'concept:name',
                                   case_id_key: str = 'case:concept:name',
                                   timestamp_key: str = 'time:timestamp',
                                   variants_cover_pct = 0.3,
                                   return_filtered_log: bool = False):
    """
    Create a Soll-Modell (planned model) as Petri Net that covers as many cases as possible with as little variants as possible.
    """
    log_for_variants = log.__deepcopy__()
    if activity_key != 'concept:name' or case_id_key != 'case:concept:name' or timestamp_key != 'time:timestamp':
        log_for_variants = log_for_variants.rename(columns={activity_key: 'concept:name',
                                                        case_id_key: 'case:concept:name',
                                                        timestamp_key: 'time:timestamp'})

    variants = pm4py.get_variants(log_for_variants)
    print(f"Total variants in entire log: {len(variants)}")

    # sort variants by the number of occurences (i.e. how many cases they represent)
    # sort in descending order with reverse=True
    sorted_variants = dict(sorted(variants.items(), key=lambda x:x[1], reverse=True))
    total_cases = sum(sorted_variants.values())
    print(f"Total cases in the entire log: {total_cases}")

    # select variants that have an accumulated occurance of x % of all cases
    sum_values = 0
    selected_variants = []
    for key, value in sorted_variants.items():
        if sum_values < total_cases*variants_cover_pct:
            sum_values = sum_values + value
            selected_variants.append(key)
    print(f"{len(selected_variants)} variants cover {variants_cover_pct*100} % of all cases")

    filtered_log = pm4py.filtering.filter_variants(log_for_variants, selected_variants)

    print(f"There are {len(filtered_log['concept:name'].unique())} events covered in filtered_log.")

    print(f"There are {len(filtered_log['case:concept:name'].unique())} cases covered in filtered_log.")

    if return_filtered_log:
        return (pm4py.discover_petri_net_inductive(filtered_log), filtered_log)
    else:
        return pm4py.discover_petri_net_inductive(filtered_log)