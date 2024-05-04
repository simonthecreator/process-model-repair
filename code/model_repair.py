import pandas as pd
import numpy as np
import pm4py
from model_repair_tree import TreeType
from pm4py.objects.petri_net.utils.align_utils import pretty_print_alignments
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo, variants as alignments_variants
from model_repair_fahland import helpers, subprocess

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def repair_event_log_to_list(alignments, moves_not_to_repair):
    repaired_log = list()
    for trace in alignments:
        repaired_trace = list()
        for move in trace:
            a1,a2 = move
            if (move in moves_not_to_repair) and a2=='>>' and a1!='>>': # not to be repaired i.e. this move (devation) to remain
                repaired_trace.append(a1) # keep event a1 in log
            elif (move not in moves_not_to_repair) and a1=='>>' and a2!='>>':
                repaired_trace.append(a2) # model move is repaired by adding missing event a2
            elif a1!='>>' and a2!='>>': # synchronous move
                 repaired_trace.append(a1) # keep event a1
        repaired_log.append(repaired_trace)
    return repaired_log

def alignment_move_to_string(move):
    assert(type(move)==tuple)
    if move[0]!=move[1]: # exclude synchronous moves
        if move[1]=='>>':
            return "log_move "+move[0]
        elif move[0]=='>>':
            return 'model_move '+move[1]
        
def get_alignment_diagnostics_per_case(log, case, net, im, fm):
    case_log = log[log['case:concept:name']==case]
    params_discounted = helpers.get_params_discounted_from_pn(net)
    alignments_diagnostics = alignments_algo.apply(case_log, net, im, fm, parameters=params_discounted, variant=alignments_variants.dijkstra_less_memory)
    #print(f"alignments_diagnostics: {alignments_diagnostics[0][0]['alignment']}")
    #alignments_diagnostics = pm4py.conformance_diagnostics_alignments(case_log, net, im, fm)
    return alignments_diagnostics[0][0]['alignment']

def remove_model_moves_invisible(alignments_per_case):
    for tupel in alignments_per_case:
        if helpers.is_silent_transition(tupel[1]): # if it is a model move for invisible transition
            alignments_per_case.remove(tupel) # remove model_move from alignments
    return alignments_per_case

def get_alignments(new_log, orig_tuple):
    old_net, old_im, old_fm = orig_tuple
    alignments = dict()
    cases = new_log['case:concept:name'].unique()
    for case in cases:
        trace_alignments = list()
        trace_alignments.append(get_alignment_diagnostics_per_case(new_log, case, old_net, old_im, old_fm))
        for trace in trace_alignments:
            trace = remove_model_moves_invisible(trace)
        alignments[case] = trace_alignments
    return alignments

def get_alignments_new(new_log, orig_tuple):
    old_net, old_im, old_fm = orig_tuple
    alignments_dict = dict()
    cases = new_log['case:concept:name'].unique()
    for case in cases:
        alignments_per_case = get_alignment_diagnostics_per_case(new_log, case, old_net, old_im, old_fm)
        alignments_per_case = remove_model_moves_invisible(alignments_per_case)
        # sort to identify equal alignments if they are are sorted differently
        alignments_per_case.sort() # sort() is in-place
        # make tuple to use as key in dict
        apc_tuple = tuple(alignments_per_case)
        if apc_tuple not in alignments_dict.keys():
            alignments_dict[apc_tuple] = [case]
        else:
            alignments_dict[apc_tuple] = alignments_dict[apc_tuple]+[case]
    return alignments_dict

def get_alignment_values_from_alignments(alignments, orig_tuple, include_move_loc = False):
    """Get alignments values which serve as feature names in one-hot-enc and decision tree graphic"""
    net, im, fm = orig_tuple
    alignment_values = list()
    for alignment_cluster in alignments.values():
        case_values = list()
        for alignment in alignment_cluster:
            for idx, tupel in enumerate(alignment):
                if tupel is not None:
                    if not helpers.is_sync_move(tupel):
                        if include_move_loc and net and im:
                            loc = subprocess.get_move_location(net, alignment, tupel, idx, im, fm)
                            # remove single quotation marks for displaying feature name as decision tree
                            loc_str = loc.__str__().replace("'", '')
                            tupel = (tupel, loc_str)
                        case_values.append(tupel)
        alignment_values.append(case_values)
    return alignment_values

def check_satisfactory_kpi(expected, satisfactory_values, typ, lower_KPI_is_better):
    if typ==TreeType.DECISION:
        return expected in satisfactory_values
    elif typ==TreeType.REGRESSION:
        if lower_KPI_is_better: # lower KPI is better e.g. throughput time -> the lower, the better
            return any(expected < val for val in satisfactory_values)
        else: # higher KPI is better, e.g. good output
            return any(expected > val for val in satisfactory_values) 

def repair_log_by_case(alignments, x, clf, satisfactory_kpi_values, typ):
    repaired_log_lists = dict()
    for case in alignments.keys():
        repaired_log_lists[case] = []
        expected_kpi_value = clf.predict(x.loc[[case]])[0]
        if check_satisfactory_kpi(expected_kpi_value, satisfactory_kpi_values, typ):
            repaired_log_lists[case] = repaired_log_lists[case]+repair_event_log_to_list(alignments[case], flatten_list(alignments[case]))
        else:
            repaired_log_lists[case] = repaired_log_lists[case]+repair_event_log_to_list(alignments[case], [])
    return repaired_log_lists

def repair_log_by_cluster(log_cluster: dict, satisfactory_kpi_values: list, alignments_per_case: dict, typ, debug, lower_KPI_is_better):
    def debug_print(s):
        if debug:
            print(s)

    repaired_log = dict()
    for rule_list, values in log_cluster.items():
        if check_satisfactory_kpi(values['pred_val'], satisfactory_kpi_values, typ, lower_KPI_is_better):
            rule_list = eval(rule_list)
            debug_print(f"rule_list {rule_list}")
            # select those features (log/model moves) that are to be included for better target value
            not_to_repair = [rule for rule in rule_list if values['impact'][rule]=='good']
            # if move+location is used as feature, then rule[1] is the actual move, otherwise it is rule
            not_to_repair = [rule[0] if type(rule[0])==tuple else rule for rule in not_to_repair ]
            debug_print(not_to_repair)
            for case in values['cases']:
                repaired_trace = list()
                alignments = alignments_per_case[case][0]
                for move in [al for al in alignments]:
                    if (move in not_to_repair) and helpers.is_log_move(move): # not to be repaired i.e. this move (devation) to remain
                        repaired_trace.append(move[0]) # keep event a1 in log
                    elif (move not in not_to_repair) and helpers.is_model_move(move):
                        repaired_trace.append(move[1]) # model move is repaired by adding missing event a2
                    elif helpers.is_sync_move(move):
                        repaired_trace.append(move[0]) # keep event a1

                repaired_log[case] = repaired_trace
    return repaired_log

def remove_wrong_clustered_cases(log_cluster: dict, target_KPI_values_per_case: dict, satisfactory_kpi_values: list):
    for values in log_cluster.values():
        for case in values['cases']:
            if any(target_KPI_values_per_case[case] < val for val in satisfactory_kpi_values):
                # wrongly clustered
                values['cases'].remove(case)
    return log_cluster

def cluster_log(x, clf):

    x_nparray = x.values.tolist()
    x_nparray = np.array(x_nparray)

    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_indicator = clf.decision_path(x_nparray)
    leaf_id = clf.apply(x_nparray)

    alignments_dict = {}

    for sample_id in range(0, x_nparray.shape[0]):
        sample_name = x.index[sample_id]
        rule_list = list()
        pred_val = clf.predict(x.loc[[sample_name]])[0]
        impact_dict = {}
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
            node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
        ]

        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                continue

            feature_name = x.columns.values[feature[node_id]]
            rule_list.append(feature_name)

            # check if value of the split feature for sample 0 is below threshold
            if x_nparray[sample_id, feature[node_id]] <= threshold[node_id]:
                threshold_sign = "<="
                impact_dict[feature_name] = 'bad'
            else:
                threshold_sign = ">"
                impact_dict[feature_name] = 'good'

        rule_list_str = rule_list.__str__()
        if rule_list_str not in alignments_dict:
            alignments_dict[rule_list_str] = {}
            alignments_dict[rule_list_str]['cases'] = [sample_name]
            alignments_dict[rule_list_str]['pred_val'] = pred_val
            alignments_dict[rule_list_str]['impact'] = impact_dict
        else:
            alignments_dict[rule_list_str]['cases'] = alignments_dict[rule_list_str]['cases']+[sample_name]

    return alignments_dict

def get_row_by_case_and_event(df, case_id, event):
    return df.loc[(df['case:concept:name']==case_id) & (df['concept:name']==event)]

def get_repaired_event_data_from_log(repaired_log_lists, new_log):
    """
    Takes the events that are in the repaired_log_lists and searches them in the new_log
    new_log is the one that is taken for possible model adaptions
    From new_log, they contain full information including timestamp and no longer only the event name
    Returns the events as dataframe
    """
    df_repaired_log = pd.DataFrame()
    for case_id in repaired_log_lists.keys():
        for trace in repaired_log_lists[case_id]:
            for event in trace:
                row_for_this_event = get_row_by_case_and_event(new_log, case_id, event)
                df_repaired_log= pd.concat([df_repaired_log, row_for_this_event]) # add row to repaired log
    return df_repaired_log

def create_log_from_repaired(repaired_log_lists):
    """
    Returns data frame with full event log data, i.e. activity_id, case_id (both already an repaired_log_lists)
    and timestamp which is generated just to keep the order of events right but it does not represent real events
    """
    assert len(repaired_log_lists) > 0, f"repaired_logs_list expected len() > 0, got: {len(repaired_log_lists)}"
    df_repaired_log = pd.DataFrame()
    for case_id, events_list in repaired_log_lists.items():
        df_case = pd.DataFrame({
            'concept:name': events_list,
            'time:timestamp': [pd.to_datetime(x) for x in range(0, len(events_list))],
            'case:concept:name': [case_id]*len(events_list)
        })
        df_repaired_log = pd.concat([df_repaired_log, df_case])
    df_repaired_log['time:timestamp'] = pd.to_datetime(df_repaired_log['time:timestamp'], format="%Y-%m-%d %H:%M:%S", utc=True)
    return df_repaired_log