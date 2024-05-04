import pm4py
from pm4py.objects.petri_net.utils.align_utils import pretty_print_alignments
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo, variants as alignments_variants
from pm4py.objects.petri_net.obj import PetriNet, Marking
import pandas as pd
from . import helpers

class AlignmentsVariants(object):
    
    def __init__(self, original_pn: PetriNet, original_im: Marking, original_fm: Marking, new_log: pd.DataFrame):
        self.original_pn=original_pn
        self.original_im=original_im
        self.original_fm=original_fm
        self.new_log=new_log
        self.variants_list = [alignments_variants.discounted_a_star, alignments_variants.state_equation_a_star,
                alignments_variants.tweaked_state_equation_a_star,  alignments_variants.dijkstra_less_memory,
                alignments_variants.dijkstra_no_heuristics,]

    def set_params_list(self):
        model_cost_dict, sync_cost_dict, trace_net_costs = helpers.create_param_cost_functions(self.original_pn)

        transition_with_higher_costs = helpers.get_transition_by_label(self.original_pn, 'e')
        model_cost_dict[transition_with_higher_costs] = 1000000

        params_discounted = {alignments_variants.discounted_a_star.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost_dict,
                  alignments_variants.discounted_a_star.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost_dict}

        params_state_eq = {alignments_variants.state_equation_a_star.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost_dict,
                  alignments_variants.state_equation_a_star.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost_dict,
                  alignments_variants.state_equation_a_star.Parameters.PARAM_TRACE_NET_COSTS: trace_net_costs}

        params_list = [params_discounted, params_state_eq, {}, {}, {}]

        return params_list
    
    def apply(self):
        params_list = self.set_params_list()
        for variant, params in zip(self.variants_list, params_list):
          print(variant.__name__)
          alignments = alignments_algo.apply(self.new_log, self.original_pn, self.original_im, self.original_fm, parameters=params, variant=variant)[0]
          print(f"Cost: {alignments[0]['cost']}")
          pretty_print_alignments(alignments)
          pm4py.view_alignments(log=self.new_log, aligned_traces=alignments)