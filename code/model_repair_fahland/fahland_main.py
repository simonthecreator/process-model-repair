from model_repair_fahland import group_into_aligned_sublogs as align_sublogs, helpers, find_loops, subprocess
from model_repair_fahland.alignments_variants_compare import AlignmentsVariants
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo, variants as alignments_variants
import pm4py
from pm4py.objects.petri_net.utils.align_utils import pretty_print_alignments
import pandas as pd
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.util import constants
from IPython.display import display
from helpers import graphics_manipulation as graph_man

class VerbosePrinter():

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def print(self, string: str):
        if self.verbose:
            print(string)

    def view_petri_net(self, pn: PetriNet, im: Marking = None, fm: Marking = None):
        if self.verbose:
            pm4py.view_petri_net(pn, im, fm)

    def pp_alignments(self, alignments):
        if self.verbose:
            pretty_print_alignments(alignments)

    def display_df(self, df: pd.DataFrame):
        if self.verbose:
            display(df)

    def print_pn_with_new_parts_marked(self, pn: PetriNet, old_pn: PetriNet):
        if self.verbose:
            pn_edited_dot = graph_man.mark_new_parts_of_pn(pn_new = pn, pn_old=old_pn)
            display(graph_man.print_png(pn_edited_dot))

def check_if_choice_struct_in_all_alignments(alignments: list, log_move: tuple, model_move: tuple, place_to_insert: PetriNet.Place):
    for ali in alignments:
        moves_list = ali['alignment']

        if (log_move in moves_list) and (model_move in moves_list):
                continue
        
        elif (log_move not in moves_list) and (model_move not in moves_list):
                continue

        else:
        # if only log_move is present, then no choice struct repair is done
            return False    
    return True

def repair_choice_struct(alignments_after_loop_repair: list, loop_net_tuple: tuple):
    """
    Implements the approach presented in `An approach to repair Petri net-based process models with choice structures` by Qi et al. from 2018
    The goal is to make more concise choice structs. It is an addition to repair technique by Fahland and Van der Aalst (`Model repair — aligning process models to reality` 2015)
    """
    loop_net, im_loop_net, fm_loop_net = loop_net_tuple
    for ali in alignments_after_loop_repair:
        to_remove_from_moves_list_idx = []
        moves_list = ali['alignment']
        break_moves_list_loop = False

        for idx, move in enumerate(moves_list):
            if helpers.is_log_move(move):
                log_move_loc = subprocess.get_move_location(net=loop_net, alignments=moves_list, move=move, move_idx=idx, initial_marking=im_loop_net, consider_model_moves=True)

                if log_move_loc:
                    place_obj = log_move_loc.most_common()[0][0]

                    if len(place_obj.out_arcs) > 0: # choice place
                        f_idx = idx # initalize further_idx
                        for further_move in moves_list[idx+1:len(moves_list)]:
                            f_idx += 1 # increase further index as if the iteration of moves_list continues

                            if helpers.is_model_move(further_move):
                                f_loc = subprocess.get_move_location(net=loop_net, alignments=moves_list, move=further_move, move_idx=f_idx, initial_marking=im_loop_net, consider_model_moves=True)
                                if f_loc:
                                    f_place_obj = f_loc.most_common()[0][0]
                                    if f_place_obj == place_obj:

                                        # make sure that the identified choice structure is in all alignments from the list
                                        # otherwise repair causes problems
                                        if check_if_choice_struct_in_all_alignments(alignments_after_loop_repair, move, further_move, place_to_insert=place_obj):

                                            # add model move to REMOVE from moves_list because it becomes redundant
                                            to_remove_from_moves_list_idx.append(f_idx)
                                            
                                            # add transition for log move to the choice place in petri net
                                            temp_log = pd.DataFrame({'concept:name': [move[0]], 'case:concept:name': ['0'], 'time:timestamp': ['1970-04-26 18:46:40']})
                                            temp_log['time:timestamp'] = pd.to_datetime(temp_log['time:timestamp'])
                                            sub_net_temp, sub_im_temp, sub_fm_temp = pm4py.discover_petri_net_ilp(temp_log)
                                            loop_net, im_loop_net, fm_loop_net = subprocess.integrate_sub_process_into_original_model(loop_net, sub_net_temp, sub_im_temp, sub_fm_temp, f_place_obj.name, set_subnet_to_next_arc=True)
                                            loop_net_tuple = (loop_net, im_loop_net, fm_loop_net)

                                            # add log move to REMOVE from moves_list because it is repaired
                                            to_remove_from_moves_list_idx.append(idx)
                                            
                                            # assume that choice can only happen once in a petri net
                                            # hence break iteration of further-moves and break iteration of moves in moves-list
                                            break_moves_list_loop = True
                                            break

            if break_moves_list_loop:
                break                        
        # remove by index to avoid other move with same transition in other location is removed
        [moves_list.pop(x) for x in to_remove_from_moves_list_idx if moves_list[x] in moves_list]

    return loop_net_tuple

def main(log_move_locations_per_trace: dict, original_pn: PetriNet, new_log: pd.DataFrame,
         avoid_flower = False, verbose: bool = False):
    printer = VerbosePrinter(verbose)

    # group into aligned sublogs
    sublogs_dict = align_sublogs.log_moves_to_sublogs_dict(log_move_locations_per_trace)
    grouped_sublogs = align_sublogs.group_into_sublogs(sublogs_dict=sublogs_dict)
    location_list_list = helpers.location_list_str_to_list(log_move_locations_per_trace)
    grouped_sublogs = align_sublogs.pick_relevant_locations(grouped_sublogs, location_list_list)

    # preparation for n_changes (see below line 175 - 178)
    # structure of grouped_sublogs: {'p1': [change0, change1], 'p2': [change0], etc.}
    # copy grouped_sublogs because it is reduced during this function
    grouped_sublogs_copy = grouped_sublogs.copy()
    changes_without_model_moves = sum([len(x) for x in grouped_sublogs_copy.values()])

    # add loops
    loop_net, sub_net = find_loops.add_loops_dict(original_pn=original_pn, sublogs=grouped_sublogs, verbose=verbose)
    im_loop_net, fm_loop_net = subprocess.create_im_fm_for_pn(loop_net)
    printer.print("loop net")
    printer.view_petri_net(loop_net, im_loop_net, fm_loop_net)

    # create alignments again
    params_discounted = helpers.get_params_discounted_from_pn(loop_net)
    alignments_after_loop_repair = alignments_algo.apply(new_log, loop_net, im_loop_net, fm_loop_net, parameters=params_discounted,
                                                    variant=alignments_variants.dijkstra_less_memory)[0]
    
    #alignments_var_obj = AlignmentsVariants(original_pn=loop_net, original_im=im_loop_net, original_fm=fm_loop_net, new_log=new_log)
    printer.print("alignments after loop repair")
    #alignments_var_obj.apply()
    
    printer.pp_alignments(alignments_after_loop_repair)

    # repair possible choice struct before subprocess-repair
    # to avoid unnecessary flower-transitions and redundant silent transition for choice structs
    loop_net, im_loop_net, fm_loop_net = repair_choice_struct(alignments_after_loop_repair=alignments_after_loop_repair,
                                                              loop_net_tuple=(loop_net, im_loop_net, fm_loop_net))

    # SUBPROCESS REPAIR
    # model move repair by adding silent transitions
    loop_net, model_move_repair_counter = subprocess.repair_model_moves(loop_net, alignments_after_loop_repair)
    # log move repair
    log_move_locations_per_trace = subprocess.get_log_move_locations_per_trace(net=loop_net,
                                                                           alignments=alignments_after_loop_repair,
                                                                           initial_marking=im_loop_net,
                                                                           consider_model_moves = True)

    sublogs_dict = align_sublogs.log_moves_to_sublogs_dict(log_move_locations_per_trace)
    grouped_sublogs = align_sublogs.group_into_sublogs(sublogs_dict=sublogs_dict)
    grouped_submodels = subprocess.get_sub_processes_for_grouped_sublogs(grouped_sublogs=grouped_sublogs)

    # print subprocesses and integrate them into original (existing) PetriNet model
    unrepaired_loop_net = loop_net.__deepcopy__()
    for loc, submodel_list in grouped_submodels.items():
        printer.print(loc)
        for submodel in submodel_list:
            sub_pn, sub_im, sub_fm = submodel
            printer.view_petri_net(sub_pn, sub_im, sub_fm)
            loc_as_list = eval(loc)
            place_name_integrate = loc_as_list[0].split(':')[0]
            printer.print(place_name_integrate)
            #if avoid_flower:
            #    integration_func = subprocess.integrate_sub_process_into_original_model_avoid_flower
            #else:
            #    integration_func = subprocess.integrate_sub_process_into_original_model
            loop_net, im_loop_net, fm_loop_net = subprocess.integrate_sub_process_into_original_model(loop_net, sub_pn, sub_im, sub_fm, place_name_integrate,
                                                                                                      avoid_flower=avoid_flower)
            printer.print("")

    printer.view_petri_net(loop_net, im_loop_net, fm_loop_net)

    # try to visualize differently (hopefully for simpler graphic model)
    format = str(constants.DEFAULT_FORMAT_GVIZ_VIEW).lower() # format is used when Variant is WO_DECORATION
    gviz = pn_visualizer.apply(loop_net, im_loop_net, fm_loop_net, variant=pn_visualizer.Variants.ALIGNMENTS,
                                parameters={"bgcolor": 'white', "decorations": None})
    #pn_visualizer.view(gviz)

    # print how many changes (repair steps) were made
    changes_total = changes_without_model_moves + model_move_repair_counter
    #print(f"# changes made: {changes_total} of which model move changes: {model_move_repair_counter}")

    return loop_net, im_loop_net, fm_loop_net, changes_total