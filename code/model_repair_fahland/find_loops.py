from . import helpers
from copy import copy
import pandas as pd
import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from typing import Optional, Tuple
from model_repair_fahland import subprocess, helpers
from model_repair_fahland.alignments_variants_compare import AlignmentsVariants
from pm4py.objects.petri_net.utils.align_utils import pretty_print_alignments
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo, variants as alignments_variants
from pm4py.objects.petri_net.utils import align_utils, check_soundness, explore_path, petri_utils, synchronous_product


def log_move_dict_to_eventlog(log_move_dict: dict) -> pd.DataFrame:
    traces_str_list = []
    for trace, moves in log_move_dict.items():
        trace_str = ''
        for move in moves:
            if trace_str != '':
                trace_str += ','
            activity = move[0]
            trace_str += activity
        traces_str_list.append(trace_str)
    return pm4py.utils.parse_event_log_string(traces_str_list)

def clustered_subtraces_to_eventlog(sublog: list) -> pd.DataFrame:
    traces_str_list = []
    for subtrace in sublog:
        subtrace = helpers.flatten_list(subtrace)
        trace_str = ''
        for activity in subtrace:
            if trace_str != '':
                trace_str += ','
            trace_str += activity
        traces_str_list.append(trace_str)
    return pm4py.utils.parse_event_log_string(traces_str_list)


def get_transitions_shortest_path(net, transition_to_populate, current_transition, transitions_shortest_path, actual_list, rec_depth,
                             max_rec_depth):
    """
    Get shortest path between transitions

    Parameters
    ----------
    net
        Petri net
    transition_to_populate
        Transition that we are populating the shortest map of
    current_transition
        Current visited transition (must explore its places)
    transitions_shortest_path
        Current dictionary
    actual_list
        Actual list of transitions to enable
    rec_depth
        Recursion depth
    max_rec_depth
        Maximum recursion depth
    """
    if rec_depth > max_rec_depth:
        return transitions_shortest_path
    if transition_to_populate not in transitions_shortest_path:
        transitions_shortest_path[transition_to_populate] = {}
    for arc in current_transition.out_arcs:
        #if arc.target.label is None:
            for arc2 in arc.target.out_arcs:
                if arc2.target not in transitions_shortest_path[transition_to_populate] or len(actual_list) + 1 < len(
                        transitions_shortest_path[transition_to_populate][arc2.target]):
                    new_actual_list = copy(actual_list)
                    new_actual_list.append(arc.target)
                    new_actual_list.append(arc2.target)
                    transitions_shortest_path[transition_to_populate][arc2.target] = copy(new_actual_list)
                    transitions_shortest_path = get_transitions_shortest_path(net, transition_to_populate, arc2.target,
                                                                    transitions_shortest_path, new_actual_list,
                                                                    rec_depth + 1, max_rec_depth)
    return transitions_shortest_path

def add_loop_transition_between_exit_and_entry(loop_transition: PetriNet.Transition, entry: list, exit: list, pn: PetriNet):
    for place in entry:
        arc_loop_to_entry = PetriNet.Arc(source=loop_transition, target=place)
        place.in_arcs.add(arc_loop_to_entry)
        loop_transition.out_arcs.add(arc_loop_to_entry)
        pn.arcs.add(arc_loop_to_entry)
    for place_name in exit:
        place_name = place_name.split(':')[0]
        place_obj = helpers.get_place_by_name(net=pn, place_name=place_name)
        if place_obj:
            arc_exit_to_loop = PetriNet.Arc(source=place_obj, target=loop_transition)
            loop_transition.in_arcs.add(arc_exit_to_loop)
            place_obj.in_arcs.add(arc_exit_to_loop)
            pn.arcs.add(arc_exit_to_loop)

def add_start_and_end_place(pn: PetriNet, entry: list, exit: list):
    # add start and end place
    start_place = PetriNet.Place('start')
    end_place = PetriNet.Place('end')
    pn.places.add(start_place)
    pn.places.add(end_place)
    # add start and end transition
    start_transition = PetriNet.Transition('start')
    end_transition = PetriNet.Transition('end')
    pn.transitions.add(start_transition)
    pn.transitions.add(end_transition)
    # add arcs
    arc_start_to_start = PetriNet.Arc(source=start_place, target=start_transition)
    start_transition.in_arcs.add(arc_start_to_start)
    start_place.out_arcs.add(arc_start_to_start)
    pn.arcs.add(arc_start_to_start)
    arc_end_to_end = PetriNet.Arc(source=end_transition, target=end_place)
    end_place.in_arcs.add(arc_end_to_end)
    end_transition.out_arcs.add(arc_end_to_end)
    pn.arcs.add(arc_end_to_end)
    # add arcs from start to entry and from exit to end
    for place in entry:
        arc_start_to_entry = PetriNet.Arc(source=start_transition, target=place)
        place.in_arcs.add(arc_start_to_entry)
        start_transition.out_arcs.add(arc_start_to_entry)
        pn.arcs.add(arc_start_to_entry)
    for place_name in exit:
        place_name = place_name.split(':')[0]
        place_obj = helpers.get_place_by_name(net=pn, place_name=place_name)
        if place_obj:
            arc_exit_to_end = PetriNet.Arc(source=place_obj, target=end_transition)
            end_transition.in_arcs.add(arc_exit_to_end)
            place_obj.out_arcs.add(arc_exit_to_end)
            pn.arcs.add(arc_exit_to_end)
    return start_place, end_place

def get_subnet_by_transitions(net: PetriNet, transitions: set) -> PetriNet:
    subnet = PetriNet()
    for t in transitions:
        subnet.transitions.add(t)
    for transition in transitions:
        for arc in transition.in_arcs:
            subnet.places.add(arc.source)
            subnet.arcs.add(arc)
        for arc in transition.out_arcs:
            subnet.places.add(arc.target)
            subnet.arcs.add(arc)

    for p in subnet.places:
        in_arcs_to_remove_from_plc = []
        out_arcs_to_remove_from_plc = []
        for arc in p.in_arcs:
            if arc.source not in transitions:
                in_arcs_to_remove_from_plc.append(arc)
        for a in in_arcs_to_remove_from_plc:
            p.in_arcs.remove(a)
    return subnet

def get_min_distance_to_entry_places(net: PetriNet, transition: PetriNet.Transition, entry_places: set) -> int:
    min_distance = float('inf')
    for arc in transition.in_arcs:
        if arc.source in entry_places:
            min_distance = min(min_distance, 1)
        else:
            min_distance = min(min_distance, get_min_distance_to_entry_places(net, arc.source, entry_places) + 1)
    return min_distance

def add_loops_dict(original_pn: PetriNet, sublogs: dict, verbose: bool = False) -> PetriNet:
    loop_net = original_pn.__deepcopy__()
    sub_net = None
    for plc, sublog in sublogs.items():
        plc = eval(plc) if type(plc) == str else plc # make sure that plc is a list, not a string that looks like a list
        loop_body_transitions = set()
        for subtrace in sublog:
            subtrace = helpers.flatten_list(subtrace)
            # iterate over only log-moves
            for move in subtrace:
                # get list of transitions that have this move as label
                transitions_for_move = helpers.get_transition_list_by_label(loop_net, move)

                if transitions_for_move: # if the move is in the current loop_net at all
                    # if there is only one transition, then add this without looping
                    if len(transitions_for_move) == 1:
                        loop_body_transitions.add(transitions_for_move[0])
                    else:
                        # if there are multiple transitions, then add the transition with the shortest distance to entry places
                        min_dist_transition = transitions_for_move[0]
                        min_dist = get_min_distance_to_entry_places(loop_net, min_dist_transition, set(plc))
                        # iterate over all transitions and find the one with the shortest distance to entry places
                        for tra in transitions_for_move:
                            # get distance to entry places for this transition
                            dist = get_min_distance_to_entry_places(loop_net, tra, set(plc))
                            if dist < min_dist:
                                min_dist_transition = tra
                                min_dist = dist
                        # add the transition with the shortest distance to entry places
                        loop_body_transitions.add(min_dist_transition)
        # in the following loop: add all transitions in between transitions in loop_body_transitions
        to_be_added_to_loop_body = []
        for trans in loop_body_transitions:
            # get paths to other transitions for each trans
            paths_to_other_trans = get_transitions_shortest_path(loop_net, 'distances', trans, {}, [], 0, 5)
            # iterate over the paths
            for other_trans, path in paths_to_other_trans['distances'].items():
                # if target of the path is also in loop_body_transitions
                if other_trans in loop_body_transitions:
                    # iterate over transitions in path (path also contains places)
                    for path_transition in [t for t in path if type(t)==PetriNet.Transition]:
                        # add the in-between transition
                        to_be_added_to_loop_body.append(path_transition)
        for t in to_be_added_to_loop_body:
            loop_body_transitions.add(t)

        loop_net_im = subprocess.create_im_fm_for_pn(loop_net)[0]
        # get entry and exit places for the loop
        entry = helpers.step_by_step_replay_to_any_place_from_set(loop_body_transitions, pn=loop_net, im=loop_net_im)

        assert entry, "Entry Marking not found in Step By Step Replay"

        # remove places from entry that are not a prefix of any transition in loop_body_transitions
        entry = helpers.remove_place_from_marking_that_is_not_prefix_of_transition(entry, loop_body_transitions)
        exit = plc
        loop_transition = PetriNet.Transition(name='loop', label='loop')
        # deepcopy to avoid side effects when the loop-transition needs to be added to both sub_net and loop_net
        loop_transition_sub_net = loop_transition.__deepcopy__()
        # sub_net is preliminary loop_net; loop_net will be changed and used later on, if the sub-net is valid
        sub_net = get_subnet_by_transitions(net=original_pn, transitions=loop_body_transitions)

        # add loop_transition inbetween exit and entry of the loop to the sub_net
        add_loop_transition_between_exit_and_entry(loop_transition=loop_transition_sub_net, entry=entry, exit=exit, pn=sub_net)
        sub_net.transitions.add(loop_transition_sub_net)

        add_start_and_end_place(sub_net, entry=entry, exit=exit)
        # get im and fm for sub_net
        sub_net_im, sub_net_fm = helpers.create_im_fm_by_in_out_arcs(sub_net)
        eventlog_from_sublog = clustered_subtraces_to_eventlog(sublog)
        # check if there are any changes by the looping, i.e. any transitions are in the loop-body -> len(body_loop_transitions) > 0
        # additinally check if all places in exit and entry are in the sub_net
        # this might not be true if a potential loop transition is not in the same location as the transition already in the original model
        # in that case, a loop would not make sense because the repeated behavior is in different locations
        exit_place_names = [place_name.split(':')[0] for place_name in exit]
        sub_net_place_names = [p.name for p in sub_net.places]
        all_exit_places_in_subnet = all([place in sub_net_place_names for place in exit_place_names])
        all_entry_places_in_subnet = all([place.name in sub_net_place_names for place in entry])
        if len(loop_body_transitions) > 0 and all_exit_places_in_subnet and all_entry_places_in_subnet and len(entry) > 0 and len(exit) > 0:
            # get alignments for the sublog and the sub_net
            alignments = alignments_algo.apply(eventlog_from_sublog, sub_net, sub_net_im, sub_net_fm,
                                             variant=alignments_variants.dijkstra_less_memory)[0]
            if alignments[0]:
                #pretty_print_alignments(alignments)
                alignment_moves = alignments[0]['alignment']
                # if there are no log moves in the alignment, then add the loop transition to the loop_net
                if all(not helpers.is_log_move(move) for move in alignment_moves):
                    if verbose:
                        print("add_loops: No log move in alignments!")
                    loop_net.transitions.add(loop_transition)
                    add_loop_transition_between_exit_and_entry(loop_transition=loop_transition, entry=entry, exit=exit, pn=loop_net)

    return loop_net, sub_net

def add_loops(original_pn: PetriNet, sublogs: dict) -> PetriNet:
    """
    OLD VERSION
    Adds loops to the original_pn based on the sublogs.
    """
    loop_net = original_pn.__deepcopy__()
    for plc, sublog in sublogs.items():
        plc = eval(plc) if type(plc) == str else plc # make sure that plc is a list, not a string that looks like a list
        loop_body_transitions = set()
        for trace_id, subtrace in sublog.items():
            # iterate over only log-moves
            for move in [log_move for log_move in subtrace if helpers.is_log_move(log_move)]:
                # get list of transitions that have this move as label
                transitions_for_move = helpers.get_transition_list_by_label(loop_net, move[0])

                if transitions_for_move: # if the move is in the current loop_net at all
                    # if there is only one transition, then add this without looping
                    if len(transitions_for_move) == 1:
                        loop_body_transitions.add(transitions_for_move[0])
                    else:
                        # if there are multiple transitions, then add the transition with the shortest distance to entry places
                        min_dist_transition = transitions_for_move[0]
                        min_dist = get_min_distance_to_entry_places(loop_net, min_dist_transition, set(plc))
                        # iterate over all transitions and find the one with the shortest distance to entry places
                        for tra in transitions_for_move:
                            # get distance to entry places for this transition
                            dist = get_min_distance_to_entry_places(loop_net, tra, set(plc))
                            if dist < min_dist:
                                min_dist_transition = tra
                                min_dist = dist
                        # add the transition with the shortest distance to entry places
                        loop_body_transitions.add(min_dist_transition)
        # in the following loop: add all transitions in between transitions in loop_body_transitions
        to_be_added_to_loop_body = []
        for trans in loop_body_transitions:
            # get paths to other transitions for each trans
            paths_to_other_trans = get_transitions_shortest_path(loop_net, 'distances', trans, {}, [], 0, 5)
            # iterate over the paths
            for other_trans, path in paths_to_other_trans['distances'].items():
                # if target of the path is also in loop_body_transitions
                if other_trans in loop_body_transitions:
                    # iterate over transitions in path (path also contains places)
                    for path_transition in [t for t in path if type(t)==PetriNet.Transition]:
                        # add the in-between transition
                        to_be_added_to_loop_body.append(path_transition)
        for t in to_be_added_to_loop_body:
            loop_body_transitions.add(t)
        transition_to_get_marking_for = helpers.get_transition_by_label(loop_net, subtrace[0][0])
        loop_net_im = subprocess.create_im_fm_for_pn(loop_net)[0]
        entry = helpers.step_by_step_replay_to_any_place_from_set(set([transition_to_get_marking_for]), pn=loop_net, im=loop_net_im)
        exit = plc
        loop_transition = PetriNet.Transition(name='loop', label='loop')
        sub_net = get_subnet_by_transitions(net=original_pn, transitions=loop_body_transitions)
        # add loop_transition inbetween exit and entry of the loop to the sub_net
        add_loop_transition_between_exit_and_entry(loop_transition=loop_transition, entry=entry, exit=exit, pn=sub_net)
        sub_net.transitions.add(loop_transition)
        add_start_and_end_place(sub_net, entry=entry, exit=exit)
        # get im and fm for sub_net
        sub_net_im, sub_net_fm = helpers.create_im_fm_by_in_out_arcs(sub_net)
        eventlog_from_sublog = log_move_dict_to_eventlog(sublog)
        model_cost_dict, sync_cost_dict, trace_net_costs = helpers.create_param_cost_functions(sub_net)
        # parameters for alignments algo
        params_discounted = helpers.get_params_discounted_from_pn(sub_net)
        alignments = alignments_algo.apply(eventlog_from_sublog, sub_net, sub_net_im, sub_net_fm, parameters=params_discounted,
                                           variant=alignments_variants.dijkstra_less_memory)
        alignment_moves = alignments[0][0]['alignment']
        if all(not helpers.is_log_move(move) for move in alignment_moves):
            print("add_loops: No log move in alignments!")
            loop_net.transitions.add(loop_transition)
            add_loop_transition_between_exit_and_entry(loop_transition=loop_transition, entry=entry, exit=exit, pn=loop_net)

    return loop_net, sub_net

def remove_infrequent_parts(pn: PetriNet, im: Marking, fm: Marking, log: pd.DataFrame, thresh: float = 0.0) -> PetriNet:
    # create new parameters for alignment-checking
    params_discounted = helpers.get_params_discounted_from_pn(pn)
    alignments = alignments_algo.apply(log, pn, im, fm, parameters=params_discounted, variant=alignments_variants.dijkstra_less_memory)[0]
    alignment_moves = alignments[0]['alignment']
    activities_list = [move[0] for move in alignment_moves]
    used_item_dict = {}
    for t in pn.transitions:
        used_item_dict[t.label] = activities_list.count(t)
    for p in pn.places:
        ingoing_transition_count = [used_item_dict[in_arc.source.label] for in_arc in p.in_arcs]
        used_item_dict[p] = sum(ingoing_transition_count)
    for key in used_item_dict.keys():
        if used_item_dict[key] < thresh:
            if not key:
                continue
            pn = helpers.remove_x_and_adjacent_arcs_from_net(key, pn)
    return pn