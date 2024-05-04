import pm4py
from typing import Optional, Tuple, List, Set
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net import semantics
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.algo.conformance.alignments.petri_net import variants as alignments_variants
from itertools import tee, islice, chain

def element_and_next(some_iterable):
    items, nexts = tee(some_iterable, 2)
    nexts = chain(islice(nexts, 1, None), [None])
    return zip(items, nexts)


def get_transition_by_label(net: PetriNet, transition_label, fired_transitions: list = None) -> Optional[PetriNet.Transition]:
    """
    Get a transition by its label
    Addition: excludes transitions that have already been fired (helpful in case there are multiple transitions with same label and they are to be iterated)

    Parameters
    ------------
    net
        Petri net
    transition_label
        Transition label
    fired_transitions
        List of (fired) transitions that are not to be returned

    Returns
    ------------
    transition
        Transition object
    """
    for t in net.transitions:
        if fired_transitions:
            if t in fired_transitions:
                continue
        if t.label == transition_label:
            return t
    return None

def get_transition_list_by_label(net: PetriNet, transition_label, fired_transitions: list = None) -> Optional[List[PetriNet.Transition]]:
    """
    Get a list of transition by the corresponding label
    Addition: excludes transitions that have already been fired (helpful in case there are multiple transitions with same label and they are to be iterated)

    Parameters
    ------------
    net
        Petri net
    transition_label
        Transition label
    fired_transitions
        List of (fired) transitions that are not to be returned

    Returns
    ------------
    List(transition)
        List of Transition objects
    """
    tranistions_list = list()
    for t in net.transitions:
        if fired_transitions:
            if t in fired_transitions:
                continue
        if t.label == transition_label:
            tranistions_list.append(t)
    if len(tranistions_list)==0:
        return None
    else:
        return tranistions_list

def get_place_by_name(net: PetriNet, place_name) -> Optional[PetriNet.Place]:
    for p in net.places:
        if p.name == place_name:
            return p
    return None

def get_next_places(place: PetriNet.Place, pn: PetriNet) -> PetriNet.Place:
    """Get next place(s) that follow the current place in the Petri net"""
    assert(place in pn.places)
    curr_marking = petri_utils.place_set_as_marking({place})
    transitions_after_curr_place = [arc.target for arc in place.out_arcs]
    for trans in transitions_after_curr_place:

        # make sure that not all places that would be populated next, are already populated (have a token)
        # this can happen if transitions have the same out-places then these places
        # are already populated after one of the trans fired
        if all([arc.target in curr_marking for arc in trans.out_arcs]):
            break

        # otherwise execute next transition until all places are populated that follow the input place
        curr_marking = semantics.execute(trans, pn=pn, m=curr_marking)

    next_places = [p for p in curr_marking]
    return next_places

def is_log_move(move: tuple) -> bool:
    return (move[0]!='>>' and move[1]=='>>')

def is_model_move(move: tuple) -> bool:
    return (move[0]=='>>' and move[1]!='>>')

def is_sync_move(move: tuple) -> bool:
    return (move[0]!='>>' and move[1]!='>>')

def is_silent_transition(trans_name: str):
    return any([trans_name.startswith(keyword) for keyword in ['silent', 'skip', 'tau', 'τ'] ])

def silent_trans_to_invisible(pn: PetriNet):
    for t in pn.transitions:
        if is_silent_transition(t.name):
            t.label = None
    return pn

def remove_substring_from_transition_labels(pn: PetriNet, sub: str = '_sub'):
    for t in pn.transitions:
        if t.label:
            if sub in t.label:
                t.label = t.label.replace(sub, '')

def get_list_intersection(l1: list, l2: list) -> list:
    intersection = []
    [intersection.append(elem) for elem in l1 if elem in l2]
    return intersection

def flatten_list(l):
    return [item for sublist in l for item in sublist]

def location_list_str_to_list(log_move_locations_per_trace: list) -> list:
    location_list_list = []
    [location_list_list.append(eval(location_list)) for location_list in log_move_locations_per_trace.keys()]
    return location_list_list

# create cost_dicts to use for parameter setting for alignments-generation
def create_param_cost_functions(petri_net: PetriNet):
    model_cost_dict = {}
    sync_cost_dict = {}
    trace_net_costs = {}
    for trans in petri_net.transitions:
        model_cost_dict[trans] = 100000
        sync_cost_dict[trans] = 0
        trace_net_costs[trans] = 10000
    return model_cost_dict, sync_cost_dict, trace_net_costs

def get_params_discounted_from_pn(pn: PetriNet):
    model_cost_dict, sync_cost_dict, trace_net_costs = create_param_cost_functions(pn)

    params_discounted = {alignments_variants.discounted_a_star.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost_dict,
           alignments_variants.discounted_a_star.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost_dict}
    return params_discounted

def transition_is_activated(t: PetriNet.Transition, marking: Marking) -> bool:
    for in_arc in t.in_arcs:
        if in_arc.source not in marking.keys():
            return False
        else:
            return True
        
def execute_silent_transition_if_activated(curr_marking: Marking, pn: PetriNet, debug=False):
    def debug_print(s):
        if debug:
            print(s)
    silent_trans_active = False
    for place in curr_marking:
        debug_print(place)
        for out_arc in place.out_arcs:
            if is_silent_transition(out_arc.target.name):
                debug_print(f"tau in out_arc.target.name")
                for in_arc in out_arc.target.in_arcs:
                    debug_print(f"in_arc: {in_arc}")
                    debug_print(curr_marking)

                if transition_is_activated(out_arc.target, curr_marking):
                    debug_print("Silent transition activated")
                    silent_trans_active = True
                    # if a silent transition is activated, execute it
                    shifted_marking = semantics.execute(out_arc.target, pn=pn, m=curr_marking)
                    if not shifted_marking:
                        debug_print(out_arc)
    if silent_trans_active:
        return shifted_marking
    else:
        return curr_marking
    
def step_by_step_token_based_replay_by_events_list(events_list: list, pn: PetriNet, im: Marking, fm: Marking, debug=False):
    def debug_print(s):
        if debug:
            print(s)
    def debug_view_petri():
        if debug:
             pm4py.view_petri_net(pn, marking, fm)   

    marking = im
    fired_transitions = []
    for event, next_event in element_and_next(events_list):
        debug_print(f"Event: {event}")
        t_obj = get_transition_by_label(pn, event, fired_transitions=fired_transitions)

        if not transition_is_activated(t_obj, marking):
            # if current event is not activated, then try to fire silent transition
            # this can happen if the very first transition after initial marking is a silent trans.
            marking = execute_silent_transition_if_activated(marking, pn)

        fired_transitions.append(t_obj)
        debug_print(f"Transition object: {t_obj}")
        marking = semantics.execute(t_obj, pn=pn, m=marking)

        if next_event: # next_event is None if the current event is the last in the list
            next_t_obj = get_transition_by_label(pn, next_event, fired_transitions=fired_transitions)
            if not transition_is_activated(next_t_obj, marking):
                # try to execute silent transition if the next trans from list is not able to fire
                marking = execute_silent_transition_if_activated(marking, pn)
        debug_view_petri()
        debug_print(marking)
        debug_print("-------------------------------")
    return marking

def step_by_step_replay_to_any_place_from_set(trs: Set[PetriNet.Transition], pn: PetriNet, im: Marking) -> Marking:
    marking = im
    # check if the intended place is activited in the current marking
    for curr_place in marking.keys():
        for out_arc in curr_place.out_arcs:
            transition_to_be_fired = out_arc.target
            if transition_to_be_fired in trs:
                return marking
    marking = semantics.execute(transition_to_be_fired, pn, marking)
    #print(f"Marking after step-by-step replay to find potential loop entry marking: {marking}")
    return marking

def remove_place_from_marking_that_is_not_prefix_of_transition(marking: Marking, t: Set[PetriNet.Transition]) -> Marking:
    places_to_remove = set()
    for place in marking.keys():
        # if any of the transitions that follows a place in the marking, is in the desired set of transitins,
        # then keep it. Remove if no target-transition is in the set of transitions "t"
        if not any([out_arc.target in t for out_arc in place.out_arcs]):
            places_to_remove.add(place)
    for place in places_to_remove:
        marking.pop(place)
    return marking

def remove_x_and_adjacent_arcs_from_net(x, pn: PetriNet) -> PetriNet:
  """Removes place x and all adjacent arcs from Petri net pn."""
  obj_x = get_transition_by_label(pn, x) # get transition-object if x is only transition-label
  if not obj_x:
      obj_x = x # assume it is a Place-object
  # remove in_arcs and adjacent nodes
  for in_arc in obj_x.in_arcs:
      src_of_adj_in_arc = in_arc.source
      src_of_adj_in_arc.out_arcs.remove(in_arc)
      pn.arcs.remove(in_arc)
  # remove out_arcs and adjacent nodes
  for out_arc in obj_x.out_arcs:
      dst_of_adj_out_arc = out_arc.target
      dst_of_adj_out_arc.in_arcs.remove(out_arc)
      pn.arcs.remove(out_arc)
  if type(obj_x)==PetriNet.Place:
      pn.places.remove(obj_x)
  elif type(obj_x)==PetriNet.Transition:
      pn.transitions.remove(obj_x)
  return pn

def deduplicate_list_of_lists(list_of_lists: List[List]) -> List[List]:
    set_of_lists_as_str = {str(lst) for lst in list_of_lists} # list is not hashable, so convert to string; set removes duplicates
    deduplicated_list = list(set_of_lists_as_str)
    dedup_list_of_lists = [eval(elem) for elem in deduplicated_list] # convert strings back to lists
    return dedup_list_of_lists

def check_if_transition_has_silent_transition_as_alternative_path(t: PetriNet.Transition, pn: PetriNet) -> bool:
    """Check if transition t has a silent transition as alternative path.
       This is to avoid multiple silent transitions that cover the same path, i.e. are redundant.

    Args:
        t (PetriNet.Transition): Transition to be checked.
        pn (PetriNet): Petri Net in which the Transition is located.

    Returns:
        bool: True if transition has silent transition as alternative path, False otherwise.
    """
    for other_t in pn.transitions:
        if is_silent_transition(other_t.name):
            incoming_places_equal = all([out_arc.target in [ot_out.target for ot_out in other_t.out_arcs] for out_arc in t.out_arcs])
            outgoing_places_equal = all([in_arc.source in [ot_in.source for ot_in in other_t.in_arcs] for in_arc in t.in_arcs])
            if incoming_places_equal and outgoing_places_equal and other_t!=t:
                return True
    return False

def add_silent_to_make_trans_optional(t: PetriNet.Transition, pn: PetriNet) -> PetriNet:
    """Add silent transition as alternative path to make transition optional. Used to repair model move.

    Args:
        t (PetriNet.Transition): Transition to be made optional.
        pn (PetriNet): Petri Net in which the Transition is located.

    Returns:
        PetriNet: Petri Net with new silent transition as alternative path to Transition t.
    """
    silent_t = PetriNet.Transition('silent_'+t.label, label='τ_'+t.label)
    pn.transitions.add(silent_t)
    for in_arc in t.in_arcs:
        source = in_arc.source
        new_in_arc = PetriNet.Arc(source, silent_t)
        source.out_arcs.add(new_in_arc)
        silent_t.in_arcs.add(new_in_arc)
        pn.arcs.add(new_in_arc)
    for out_arc in t.out_arcs:
        target = out_arc.target
        new_out_arc = PetriNet.Arc(silent_t, target)
        target.in_arcs.add(new_out_arc)
        silent_t.out_arcs.add(out_arc)
        pn.arcs.add(new_out_arc)
    return pn

def create_im_fm_by_in_out_arcs(pn: PetriNet) -> Tuple[Marking, Marking]:
    """create initial marking and final marking for given Petri Net
       Places that have no incoming arcs are set as initial marking
       Place without outgoing arcs are set as final marking

    Args:
        pn (PetriNet): Petri Net to create initial and final marking for

    Returns:
        Tuple[Marking, Marking]: Initial and Final Marking
    """
    source_places = set()
    sink_places = set()
    for plc in pn.places:
        if len(plc.in_arcs)==0: # this place has no incoming arcs
            source_places.add(plc)
        if len(plc.out_arcs)==0:
            sink_places.add(plc)

    ini_marking = petri_utils.place_set_as_marking(source_places)
    fin_marking = petri_utils.place_set_as_marking(sink_places)

    return ini_marking, fin_marking