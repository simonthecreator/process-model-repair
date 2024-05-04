import pm4py
from . import helpers
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net import semantics
from typing import Tuple
import random

def get_move_location(net: PetriNet, alignments: list, move: tuple, move_idx: int, initial_marking: Marking, consider_model_moves: bool):
    prev_moves_list = []
    # for all moves that are before current move
    for prev_move in alignments[0:move_idx]:
        # only add sync moves in prev_moves_list
        if helpers.is_sync_move(prev_move):
            prev_moves_list.append(prev_move)
        if consider_model_moves:
            # also add model moves in prev_moves_list
            if helpers.is_model_move(prev_move):
                prev_moves_list.append(prev_move)
    # start with initial marking
    marking = initial_marking
    for t in prev_moves_list:
        t_label = t[1]
        # get transition object by label
        t_obj = helpers.get_transition_by_label(net=net, transition_label=t_label)
        if marking:
            # update marking after execution of transition
            marking = semantics.execute(t_obj, pn=net, m=marking)
    return marking

def get_log_move_location(net: PetriNet, alignments: list, move: tuple, move_idx: int, initial_marking: Marking, consider_model_moves: bool):
    if helpers.is_log_move(move):
        return get_move_location(net, alignments, move, move_idx, initial_marking, consider_model_moves)
    else:
        return None
    
def get_log_move_locations_per_trace(net: PetriNet, alignments: list, initial_marking: Marking, consider_model_moves: bool = True):
    location_dict = dict()
    trace_nr = 1
    for algn_trace in alignments:
        trace_nr_str = 'trace '+str(trace_nr)
        # for move in the alignments of this trace
        for idx, move in enumerate(algn_trace['alignment']):
            if helpers.is_log_move(move):
                # get location of move
                location = get_log_move_location(net=net, alignments=algn_trace['alignment'], move=move, move_idx = idx, initial_marking=initial_marking, consider_model_moves=consider_model_moves)
                if location is None:
                    continue
                # make location string to use it as dict-key
                location_str = location.__str__()
                # add location to dict
                if location_str not in location_dict:
                    location_dict[location_str] = {}
                # add trace_nr to dict    
                if trace_nr_str not in location_dict[location_str]:
                    location_dict[location_str][trace_nr_str] = []
                # add move to dict
                location_dict[location_str][trace_nr_str].append(move)
        trace_nr += 1
    return location_dict

def repair_model_moves(net: PetriNet, alignments: list, debug=False) -> PetriNet:
    def debug_print(s):
        if debug:
            print(s)

    counter = 0
    repaired_model_moves = set()
    for algn_trace in alignments:
        model_moves = [move for move in algn_trace['alignment'] if helpers.is_model_move(move)]
        for move in model_moves:
            if move[1] is not None:
                if not helpers.is_silent_transition(move[1]) and move[1] not in repaired_model_moves:
                    debug_print(move)
                    transition_to_be_optional = helpers.get_transition_by_label(net, move[1])
                    if not helpers.check_if_transition_has_silent_transition_as_alternative_path(transition_to_be_optional, net):
                        # no silent transition as alternative path already so add silent transition
                        net = helpers.add_silent_to_make_trans_optional(transition_to_be_optional, net)
                        counter += 1
                    repaired_model_moves.add(move[1])
                    debug_print(repaired_model_moves)
    return net, counter

def get_optimal_location_from_list(location_list_list):
    return_marking = location_list_list[0][-1]
    if len(location_list_list) == 2:
        intersec = helpers.get_list_intersection(location_list_list[0], location_list_list[1])
        if len(intersec) > 0:
          return_marking = intersec[-1]
    # convert Marking to place_name as string
    place_name = return_marking.split(':')[0]
    return place_name

def get_sub_process(log_move_locations_per_trace: dict) -> PetriNet:
    trace_list = list()   
    for location in log_move_locations_per_trace:
        for trace, events in log_move_locations_per_trace[location].items():
            events_for_this_trace = ''
            for event in events:
                activity_id = event[0]
                if events_for_this_trace != '':
                    events_for_this_trace += ','
                events_for_this_trace += activity_id
            trace_list.append(events_for_this_trace)
    temp_log = pm4py.utils.parse_event_log_string(trace_list)
    temp_pn, temp_im, temp_fm = pm4py.discover_petri_net_ilp(temp_log)
    return temp_pn, temp_im, temp_fm

def get_sub_processes_for_grouped_sublogs(grouped_sublogs: dict) -> PetriNet:
    grouped_submodels = dict()
    for location, grouped_sublogs in grouped_sublogs.items():
        location_models = list()
        for group in grouped_sublogs:
            trace_list = list()
            for trace in group:
                events_for_this_trace = ''
                for activity in trace:
                    if events_for_this_trace != '':
                        events_for_this_trace += ','
                    events_for_this_trace += activity
                trace_list.append(events_for_this_trace)
            # create petri net for each group of traces
            temp_log = pm4py.utils.parse_event_log_string(trace_list)
            temp_pn, temp_im, temp_fm = pm4py.discover_petri_net_ilp(temp_log)
            
            # change labels to distinguish from events with same labels that are already in original model
            for t in temp_pn.transitions:
                if t.label is None:
                    continue
                t.label = t.label+'_sub'

            location_models.append((temp_pn, temp_im, temp_fm))
        grouped_submodels[location] = location_models
    return grouped_submodels

def fix_submarking_and_asstrans(pn: PetriNet) -> None:
    """
    overwrite sub_marking for each transition with the pre_set
    and overwrite ass_trans for each place in sub_marking with post_set of the place
    This sub_marking and pre_set are usually the same. This is also true for ass_trans and post_set (they are equal)
    The reason is probably a bug in pm4py. This function just copies values to other places to make the alignments work later on
    because alignments rely on sub_marking and ass_trans
    (Stand: 28.04.2023, pm4py-Version 2.7.3)
    """
    for t in pn.transitions:
      # set sub_marking to pre_set
      t.sub_marking = petri_utils.pre_set(t)
      for p in t.sub_marking:
          # set ass_trans ("associated transitions") to place's post-set
          p.ass_trans = petri_utils.post_set(p)

def remove_old_in_and_out_arcs(pn: PetriNet) -> None:
    """
    Due to the integration of the sub-process in the original process, there are in and out-going arcs left that point to or from places that are no longer in the newly created, integrated model
    This should usually be the old source and sink place in the sub-process as they are not integrated into the new model
    TODO: This probably can be done more efficiently and elegant alongside the actual integration of the places and transitins into the new model and not afterwards using this very function

    Args:
        pn (PetriNet): Petri Net where old in and out arcs must be removed for transitions
    """
    for tra in pn.transitions:
        # create lists because elements cannot be removed from a set during looping over this set
        in_arcs_to_be_removed = []
        out_arcs_to_be_removed = []

        for in_arc in tra.in_arcs:
            # if the source-place is no longer in the model, add to to-be-deleted list
            if (in_arc.source not in pn.places):
                in_arcs_to_be_removed.append(in_arc)

        for out_arc in tra.out_arcs:
            # if target-place is no longer in the model, add to to-be-deleted list
            if out_arc.target not in pn.places:
                out_arcs_to_be_removed.append(out_arc)

        # delete arcs that point from or to non-existing places via list
        for arc in in_arcs_to_be_removed:
            tra.in_arcs.remove(arc)
        for arc in out_arcs_to_be_removed:
            tra.out_arcs.remove(arc)

def create_im_fm_for_pn(pn: PetriNet) -> Tuple[Marking, Marking]:
    """create initial marking and final marking for given Petri Net
       Precondition: places with name 'source' and 'sink' exist

    Args:
        pn (PetriNet): Petri Net to create initial and final marking for

    Returns:
        Tuple[Marking, Marking]: Initial and Final Marking
    """
    source_place = helpers.get_place_by_name(pn, 'source')
    ini_marking = petri_utils.place_set_as_marking({source_place})

    sink_place = helpers.get_place_by_name(pn, 'sink')
    fin_marking = petri_utils.place_set_as_marking({sink_place})
    return ini_marking, fin_marking

def change_place_names_if_duplicates(pn: PetriNet, pn2: PetriNet):
    """checks if places have same names in two PetriNets. If so, rename in the first PetriNet (concatenate string 'x')

    Args:
        pn (PetriNet): _description_
        pn2 (PetriNet): _description_
    """
    for pl in pn.places:
        if pl.name in [pl_pn2.name for pl_pn2 in pn2.places]:
            if pl.name != 'source' and pl.name != 'sink':
                upper_bound = len(pn.places) + 100
                suffix_to_make_unique = "_" + str(random.randrange(0, upper_bound))
                while pl.name + suffix_to_make_unique in [place.name for place in pn.places]:
                    # make sure that new place-name with suffix is not randomly already used as place-name
                    suffix_to_make_unique = "_" + str(random.randrange(0, upper_bound))
                pl.name = pl.name + suffix_to_make_unique

def integrate_sub_process_into_original_model(original_model: PetriNet, sub_model: PetriNet, sub_im: Marking, sub_fm: Marking, place_name_to_integrate: str, avoid_flower: bool = False, set_subnet_to_next_arc: bool = False) -> Tuple[PetriNet, Marking, Marking]:
    sub_model_src_place = sub_im.most_common()[0][0]
    sub_model_sink_place = sub_fm.most_common()[0][0]
    repaired_pn = original_model.__deepcopy__()
    start_integration = helpers.get_place_by_name(repaired_pn, place_name_to_integrate)

    change_place_names_if_duplicates(repaired_pn, sub_model)

    if avoid_flower: # ------ AVOID FLOWER ------ 

        # add all places except source place of sub-model
        for place in sub_model.places:
            if place != sub_model_src_place:
                repaired_pn.places.add(place)

        # add all transitions from sub-model
        for transition in sub_model.transitions:
            repaired_pn.transitions.add(transition)

        arcs_to_be_removed = set()
        for arc in start_integration.out_arcs:
            new_arc = PetriNet.Arc(source=sub_model_sink_place, target=arc.target)
            sub_model_sink_place.out_arcs.add(new_arc)
            repaired_pn.arcs.add(new_arc)
            arcs_to_be_removed.add(arc)
            # add new_arc as in-arc to target
            arc.target.in_arcs.add(new_arc)
            arc.target.in_arcs.remove(arc)
        for a in arcs_to_be_removed:
            repaired_pn.arcs.remove(a)
            start_integration.out_arcs.remove(a)

        new_t_name = sub_model.transitions.pop().name # pop() returns first element in set
        new_t = PetriNet.Transition(name='skip_'+new_t_name, label='τ_x')
        t_in_arc = PetriNet.Arc(source=start_integration, target=new_t)
        t_out_arc = PetriNet.Arc(source=new_t, target=sub_model_sink_place)
        new_t.in_arcs.add(t_in_arc)
        new_t.out_arcs.add(t_out_arc)
        start_integration.out_arcs.add(t_in_arc)
        sub_model_sink_place.in_arcs.add(t_out_arc)
        repaired_pn.transitions.add(new_t)
        repaired_pn.arcs.add(t_in_arc)
        repaired_pn.arcs.add(t_out_arc)

        # add arcs from sub-model
        for arc in sub_model.arcs:
            if arc.source == sub_model_src_place:
                # replace arc where source-place of sub-model is source with arc where entry-place for the sub-model is source
                new_arc = PetriNet.Arc(source=start_integration, target=arc.target)
                repaired_pn.arcs.add(new_arc)
                # add arc where sub-model starts to out-arcs entry-place and vice-versa to in-arcs of respective target-place
                start_integration.out_arcs.add(new_arc)
                arc.target.in_arcs.add(new_arc)
            else:
                repaired_pn.arcs.add(arc)

        # fix places that are wrongly names 'sink' due to integrating sub_model to repaired_pn
        # get the highest number that is in place-names, e.g. 9 if from p_9
        places_exluding_sink_source = [p for p in repaired_pn.places if p.name not in ['sink', 'source']]
        max_place_nr = max([int(p.name[-1]) for p in places_exluding_sink_source])
        
        for p in repaired_pn.places:
            if len(p.out_arcs) > 0:
                if p.name == 'sink': # this place is wrongly named 'sink'
                    max_place_nr += 1
                    p.name = 'p_' + str(max_place_nr)

        fix_submarking_and_asstrans(repaired_pn)
        remove_old_in_and_out_arcs(repaired_pn)
        im, fm = helpers.create_im_fm_by_in_out_arcs(repaired_pn)            

    else: # ------ FAHLAND TECHNIQUE WITHOUT AVOID FLOWER ------ 

        if not set_subnet_to_next_arc:
            end_integration = start_integration
        else:
            arc_to_t = list(start_integration.out_arcs)[0]
            next_t = arc_to_t.target
            arc_to_p = list(next_t.out_arcs)[0]
            end_integration = arc_to_p.target

        # add all places except source and sink place of sub-model
        for place in sub_model.places:
            if place != sub_model_src_place and place != sub_model_sink_place:
                repaired_pn.places.add(place)

        # add all transitions from sub-model
        for transition in sub_model.transitions:
            repaired_pn.transitions.add(transition)

        # add arcs from sub-model
        for arc in sub_model.arcs:
            if arc.source == sub_model_src_place:
                # replace arc where source-place of sub-model is source with arc where entry-place for the sub-model is source
                new_arc = PetriNet.Arc(source=start_integration, target=arc.target)
                repaired_pn.arcs.add(new_arc)
                # add arc where sub-model starts to out-arcs entry-place and vice-versa to in-arcs of respective target-place
                start_integration.out_arcs.add(new_arc)
                arc.target.in_arcs.add(new_arc)
            elif arc.target == sub_model_sink_place:
                # replace arc where sink-place of sub-model is target with arc where entry-place of the sub-model is target
                new_arc = PetriNet.Arc(source=arc.source, target=end_integration)
                repaired_pn.arcs.add(new_arc)
                # add arc where sub-model ends to in-arcs of entry-place and vice-versa to out-arcs of respective source-place
                end_integration.in_arcs.add(new_arc)
                arc.source.out_arcs.add(new_arc)
            else:
                repaired_pn.arcs.add(arc)
    
        fix_submarking_and_asstrans(repaired_pn)
        remove_old_in_and_out_arcs(repaired_pn)
        im, fm = create_im_fm_for_pn(repaired_pn)

    return repaired_pn, im, fm

def integrate_sub_process_into_original_model_avoid_flower(original_model: PetriNet, sub_model: PetriNet, sub_im: Marking, sub_fm: Marking, place_name_to_integrate: str) -> Tuple[PetriNet, Marking, Marking]:
    sub_model_src_place = sub_im.most_common()[0][0]
    sub_model_sink_place = sub_fm.most_common()[0][0]
    repaired_pn = original_model.__deepcopy__()
    start_integration = helpers.get_place_by_name(repaired_pn, place_name_to_integrate)

    change_place_names_if_duplicates(repaired_pn, sub_model)

    # add all places except source place of sub-model
    for place in sub_model.places:
      if place != sub_model_src_place:
        repaired_pn.places.add(place)

    # add all transitions from sub-model
    for transition in sub_model.transitions:
      repaired_pn.transitions.add(transition)

    arcs_to_be_removed = set()
    for arc in start_integration.out_arcs:
        new_arc = PetriNet.Arc(source=sub_model_sink_place, target=arc.target)
        sub_model_sink_place.out_arcs.add(new_arc)
        repaired_pn.arcs.add(new_arc)
        arcs_to_be_removed.add(arc)
        # add new_arc as in-arc to target
        arc.target.in_arcs.add(new_arc)
        arc.target.in_arcs.remove(arc)
    for a in arcs_to_be_removed:
        repaired_pn.arcs.remove(a)
        start_integration.out_arcs.remove(a)

    new_t = PetriNet.Transition(name='skip_x', label='τ_x')
    t_in_arc = PetriNet.Arc(source=start_integration, target=new_t)
    t_out_arc = PetriNet.Arc(source=new_t, target=sub_model_sink_place)
    new_t.in_arcs.add(t_in_arc)
    new_t.out_arcs.add(t_out_arc)
    start_integration.out_arcs.add(t_in_arc)
    sub_model_sink_place.in_arcs.add(t_out_arc)
    repaired_pn.transitions.add(new_t)
    repaired_pn.arcs.add(t_in_arc)
    repaired_pn.arcs.add(t_out_arc)

    # add arcs from sub-model
    for arc in sub_model.arcs:
      if arc.source == sub_model_src_place:
          # replace arc where source-place of sub-model is source with arc where entry-place for the sub-model is source
          new_arc = PetriNet.Arc(source=start_integration, target=arc.target)
          repaired_pn.arcs.add(new_arc)
          # add arc where sub-model starts to out-arcs entry-place and vice-versa to in-arcs of respective target-place
          start_integration.out_arcs.add(new_arc)
          arc.target.in_arcs.add(new_arc)
      else:
          repaired_pn.arcs.add(arc)

    fix_submarking_and_asstrans(repaired_pn)
    remove_old_in_and_out_arcs(repaired_pn)
    im, fm = helpers.create_im_fm_by_in_out_arcs(repaired_pn)

    return repaired_pn, im, fm