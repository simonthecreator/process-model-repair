import pandas as pd
import pm4py
import model_repair as mr
from model_repair_tree import TreeType, TreeHandler
from helpers import tree_edit_distance
from sklearn.model_selection import train_test_split
from IPython.display import display
import numpy as np
import pathlib
import os
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.evaluation.simplicity import algorithm as simplicity
from pm4py.algo.evaluation.precision import algorithm as precision, variants as prec_variants
from pm4py.algo.evaluation.generalization import algorithm as generalization, variants as gen_variants
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.convert import convert_to_petri_net as pt_to_petri_factory
from model_repair_fahland import subprocess, helpers, fahland_main as fahland
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo, variants as alignments_variants

class MainRepair:
    def __init__(self, alternative_log: pd.DataFrame, original_net: PetriNet,
                 original_im: Marking, original_fm: Marking,
                 target_KPI_values_per_case: dict(),
                 satisfactory_values: list, test_size: float = 0.33,
                 output_dir: pathlib.Path = None,
                 debug: bool = False,
                 run_in_ipynb: bool = True,
                 lower_KPI_is_better: bool = False) -> None:
        self.alternative_log = alternative_log
        self.original_net = original_net
        self.original_im = original_im
        self.original_fm = original_fm
        self.target_KPI_values_per_case = target_KPI_values_per_case
        self.satisfactory_values = satisfactory_values
        self.output_dir = output_dir
        self.test_size = test_size
        self.debug = debug
        self.run_in_ipynb = run_in_ipynb
        self.lower_KPI_is_better = lower_KPI_is_better

    def debug_print(self, s):
        if self.debug:
            print(s)

    def save_petri_net(self, net: PetriNet, im: Marking, fm: Marking, file_name: str):
        if self.output_dir:
            if os.path.exists(self.output_dir):
                pm4py.save_vis_petri_net(net, im, fm,
                                    file_path = os.path.join(self.output_dir, file_name))
            else:
                print("Tried to save Petri Net but path does not exist.")

    def main(self):

        # save original petri net
        self.save_petri_net(self.original_net, self.original_im, self.original_fm, "original.png")

        self.printer = fahland.VerbosePrinter(verbose=self.run_in_ipynb)

        self.create_train_test_split()
        self.X_train_log = self.alternative_log[self.alternative_log['case:concept:name'].isin(self.X_train)]

        self.X_test_log = self.alternative_log[self.alternative_log['case:concept:name'].isin(self.X_test)]
        if len(self.satisfactory_values)==1:
          print(f"Number of cases below satisfactory threshold: {len([y for y in self.y_train if y<=self.satisfactory_values[0]])} of {len(self.y_train)} total cases in training data.")
          print(f"Number of cases below satisfactory threshold: {len([y for y in self.y_test if y<=self.satisfactory_values[0]])} of {len(self.y_test)} total cases in test data.")
        self.create_original_conformance()

        self.alignments, self.alignment_values, self.alignment_values_loc = self.get_alignments()
        
        print_tree = False
        if self.lower_KPI_is_better:
            print_tree = False
            
        self.repaired_log_df = self.get_repaired_log_df(TreeType.REGRESSION,
                                                        alignment_values=self.alignment_values,
                                                        print_tree=print_tree)
        self.repaired_log_df_loc = self.get_repaired_log_df(TreeType.REGRESSION,
                                                        alignment_values=self.alignment_values_loc,
                                                        print_tree=print_tree)

        # create petri nets from repaired_log
        self.create_petri_nets()
        self.print_petri_nets()

        self.add_other_conformances()

    def create_train_test_split(self):
        """"
        Training data is used to identify log and model moves and repair the model based on the respective alignments.
        Traces in test data are used to split which fit with the original model and which fit with the repaired model.
        The KPI values of these two groups are compared to see whether the repaired model is advantageous for future logs.
        """
        # the following line can be used when we want to use only cases that are not already conformant to the original model
        # then 'alternative_log_cases' must be used instead of 'case_ids' in this function
        #alternative_log_cases = self.get_alternative_log_cases()
        self.case_ids = self.alternative_log['case:concept:name'].unique()
        self.target_kpi_values = [self.target_KPI_values_per_case[case] for case in self.case_ids]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.case_ids, self.target_kpi_values, test_size=self.test_size, random_state=42)
        
    def get_alternative_log_cases(self):
        """Returns a list of those cases that are not compliant to the original model"""
        non_conf_with_original = self.exclude_cases_conformant_with_original_model()
        return list(non_conf_with_original['case:concept:name'].unique())
    
    def exclude_cases_conformant_with_original_model(self):
        case_ids = self.alternative_log['case:concept:name'].unique()
        case_fitness = []
        for case_id in case_ids:
            case_log = self.alternative_log[self.alternative_log['case:concept:name']==case_id]
            tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(case_log, self.original_net, self.original_im, self.original_fm)
            case_fitness.append(tbr_diagnostics[0]['trace_is_fit'])
        alternative_conformant_with_original_model = pd.DataFrame({'case:concept:name': case_ids,
                                                                'trace_is_fit': case_fitness})
        conformant_cases = alternative_conformant_with_original_model[alternative_conformant_with_original_model['trace_is_fit']]
        non_conf_with_original = self.alternative_log[~self.alternative_log['case:concept:name'].isin(conformant_cases['case:concept:name'])]
        return non_conf_with_original

    def create_original_conformance(self):
        X_test_cases_df = pd.DataFrame(self.X_test, columns=['case:concept:name'])

        alignment_diagnostics = pm4py.conformance_diagnostics_alignments(self.X_test_log, self.original_net, self.original_im, self.original_fm,
                                        activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')[0]
        
        tbr_diagnostics_original = pm4py.conformance_diagnostics_token_based_replay(self.X_test_log, self.original_net, self.original_im, self.original_fm,
                                                                          activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')
        
        conformant_alignments_original = []
        for diag in alignment_diagnostics:
            if diag:
                conformant_alignments_original.append(diag['fitness']==1)
            else:
                conformant_alignments_original.append(False)
        X_test_cases_df['conformant_alignments_original'] = pd.Series(conformant_alignments_original)

        X_test_cases_df['conformant_original'] = pd.Series([diag['trace_is_fit'] for diag in tbr_diagnostics_original])
        self.X_test_cases_df = X_test_cases_df
    
    def add_other_conformances(self):
        for result_petri, result_name in zip([(self.repaired_net_IM, self.repaired_im_IM, self.repaired_fm_IM),
                                              (self.repaired_net_HEU, self.repaired_im_HEU, self.repaired_fm_HEU),
                             (self.kpi_based_pn, self.kpi_based_im, self.kpi_based_fm),
                             (self.avoid_flower_pn, self.avoid_flower_im, self.avoid_flower_fm),
                             (self.loc_pn, self.loc_im, self.loc_fm),
                             (self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm),
                             #(self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm)
                             ],
                             ['repaired_IM', 'repaired_HEU',
                              'kpi_based', 'avoid_flower', 'move_loc', 'repair_all_traces'#, 'repair_all_traces_train_cases'
                              ]):
            alignment_diagnostics = pm4py.conformance_diagnostics_alignments(self.X_test_log, result_petri[0], result_petri[1], result_petri[2],
                                                            activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')[0]
            
            tbr_diagnostics = pm4py.conformance_diagnostics_token_based_replay(self.X_test_log, result_petri[0], result_petri[1], result_petri[2],
                                                                   activity_key='concept:name', case_id_key='case:concept:name', timestamp_key='time:timestamp')


            conformant_alignments_original = []
            for diag in alignment_diagnostics:
                if diag:
                    conformant_alignments_original.append(diag['fitness']==1)
                else:
                    conformant_alignments_original.append(False)
            self.X_test_cases_df['conformant_alignments_'+result_name] = pd.Series(conformant_alignments_original)
            self.X_test_cases_df['conformant_'+result_name] = pd.Series([diag['trace_is_fit'] for diag in tbr_diagnostics])

    def get_alignments(self):
        self.set_transition_label_to_name(self.original_net)
        orig_tuple = (self.original_net, self.original_im, self.original_fm)
        alignments = mr.get_alignments(self.X_train_log, orig_tuple)
        self.alignments_new = mr.get_alignments_new(self.X_train_log, orig_tuple)
        alignment_values = mr.get_alignment_values_from_alignments(alignments, orig_tuple, include_move_loc=False)
        alignment_values_loc = mr.get_alignment_values_from_alignments(alignments, orig_tuple, include_move_loc=True)
        return alignments, alignment_values, alignment_values_loc

    def set_transition_label_to_name(self, net):
        for t in net.transitions:
          if t.label==None:
              t.label = t.name

    def create_decision_tree(self, typ: TreeType, print_tree: bool, alignment_values: list):
        tree_handler = TreeHandler(alignment_values, typ=typ)
        self.one_hot_enc = tree_handler.prepare_one_hot(self.alignments.keys())
        clf_tree = tree_handler.create_tree(target_values=self.y_train,
                              case_ids=self.alignments.keys())
        if print_tree:
            graph = tree_handler.tree_to_graph()
            display(graph)

        # CROSS VALIDATION of Regression Tree
        # from sklearn import tree
        # from sklearn.tree import DecisionTreeRegressor
        # from sklearn.model_selection import cross_val_score
        # reshaped_X = self.case_ids.reshape(-1, 1)

        # depth = []
        # for i in range(3,20):
        #     other_clf = DecisionTreeRegressor(max_depth=i, random_state = 0)
        #     # Perform 7-fold cross validation 
        #     scores = cross_val_score(estimator=other_clf, X=reshaped_X, y=self.target_kpi_values, cv=5, n_jobs=4)
        #     depth.append((i,scores.mean()))
        #     print((i,scores.mean()))
        
        # scores = cross_val_score(estimator=clf_tree, X=reshaped_X, y=self.target_kpi_values, cv=5, n_jobs=4)
        # print(f"CROSS-VALIDATION Scores: {scores}")
        # print(f"CROSS-VALIDATION - Mean Score: {scores.mean()}")    
        return clf_tree
    
    def filter_alignments_frequent(self, min_freq: int = 0):
        # add case ids to one_hot_enc
        one_hot_case_id = self.one_hot_enc.reset_index()
        # group by all columns except 'case' and count the number of cases
        grouped = one_hot_case_id.groupby(by=one_hot_case_id.columns.to_list()[1:])['case'].apply(list).reset_index()
        grouped['n_cases'] = grouped['case'].apply(lambda x: len(x))
        grouped = grouped[['n_cases', 'case']]
        # filter alignments with less than min_freq cases
        grouped = grouped[grouped['n_cases'] > min_freq].sort_values(by=['n_cases'], ascending=False)
        # get the case ids of the filtered alignments
        list_case_ids = mr.flatten_list(grouped['case'])
        # filter the alignments
        filtered_alignments_dict = {key: self.alignments[key] for key in list_case_ids}
        return filtered_alignments_dict

    def filter_alignments_frequent_new(self, log_cluster: dict, min_freq: int = 0):
        filtered_dict = {key: val for key, val in log_cluster.items() if len(val['cases']) >= min_freq}
        return filtered_dict

    def get_repaired_log_df(self, typ:TreeType, alignment_values: list, print_tree: bool = False):
        clf_tree = self.create_decision_tree(typ, print_tree, alignment_values)
        log_cluster = mr.cluster_log(self.one_hot_enc, clf_tree)
        # filter infrequent cases (by default don't filter)
        log_cluster = self.filter_alignments_frequent_new(log_cluster)
        # remove those cases where the actual KPI value is not satisfactory
        mr.remove_wrong_clustered_cases(log_cluster, target_KPI_values_per_case=self.target_KPI_values_per_case, satisfactory_kpi_values=self.satisfactory_values)
        repaired_log_by_cluster = mr.repair_log_by_cluster(log_cluster, satisfactory_kpi_values=self.satisfactory_values,
                                 alignments_per_case=self.alignments, typ=typ, debug=self.debug,
                                 lower_KPI_is_better = self.lower_KPI_is_better)
        self.debug_print("REPAIRED BY CLUSTER")
        [self.debug_print((key, val)) for key, val in repaired_log_by_cluster.items()]
        return mr.create_log_from_repaired(repaired_log_by_cluster)

    def set_transitions_invisible_and_replayable(self, pn: PetriNet):
        # set tau transitions to invisible in order to make net replayable
        helpers.silent_trans_to_invisible(pn)
        # and remove substrings from transition labels that were used for graphic manipulation to make net replayable
        helpers.remove_substring_from_transition_labels(pn)

    def print_petri_nets(self):

        self.printer.print("\nKPI-based Technique")
        self.printer.print_pn_with_new_parts_marked(self.kpi_based_pn, old_pn=self.original_net)
        self.set_transitions_invisible_and_replayable(self.kpi_based_pn)
        self.printer.view_petri_net(self.kpi_based_pn, self.kpi_based_im, self.kpi_based_fm)
        self.save_petri_net(self.kpi_based_pn, self.kpi_based_im, self.kpi_based_fm, "kpi_based.png")

        self.printer.print("\nAVOID FLOWER")
        self.printer.print_pn_with_new_parts_marked(self.avoid_flower_pn, old_pn=self.original_net)
        self.set_transitions_invisible_and_replayable(self.avoid_flower_pn)
        self.printer.view_petri_net(self.avoid_flower_pn, self.avoid_flower_im, self.avoid_flower_fm)
        self.save_petri_net(self.avoid_flower_pn, self.avoid_flower_im, self.avoid_flower_fm, "avoid_flower.png")

        self.printer.print("\nKPI-based with move+location as feature")
        self.printer.print_pn_with_new_parts_marked(self.loc_pn, old_pn=self.original_net)
        self.set_transitions_invisible_and_replayable(self.loc_pn)
        self.printer.view_petri_net(self.loc_pn, self.loc_im, self.loc_fm)
        self.save_petri_net(self.loc_pn, self.loc_im, self.loc_fm, "move_loc.png")   

        self.printer.print("\nRepair all Traces")
        self.printer.print_pn_with_new_parts_marked(self.repair_all_traces_pn, old_pn=self.original_net)
        self.set_transitions_invisible_and_replayable(self.repair_all_traces_pn)
        self.printer.view_petri_net(self.repair_all_traces_pn)
        self.save_petri_net(self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm, "repair_all.png")

        self.printer.print("\nRepaired Net Inductive Miner")
        self.set_transitions_invisible_and_replayable(self.repaired_net_IM)
        self.printer.view_petri_net(self.repaired_net_IM)
        self.save_petri_net(self.repaired_net_IM, self.repaired_im_IM, self.repaired_fm_IM, "IM.png")
        
        self.printer.print("Repaired Net Heuristic Miner")
        self.set_transitions_invisible_and_replayable(self.repaired_net_HEU)
        self.printer.view_petri_net(self.repaired_net_HEU)
        self.save_petri_net(self.repaired_net_HEU, self.repaired_im_HEU, self.repaired_fm_HEU, "HEU.png")

    def create_alignments_from_repaired_log(self, repaired_log_df: pd.DataFrame, parameters: dict):
        return alignments_algo.apply(repaired_log_df,
                                     self.original_net,
                                     self.original_im, self.original_fm,
                                     parameters=parameters,
                                     variant=alignments_variants.dijkstra_less_memory)[0]  

    def create_petri_nets(self) -> None:
        self.repaired_net_IM, self.repaired_im_IM, self.repaired_fm_IM = pm4py.discover_petri_net_inductive(self.repaired_log_df)
        self.repaired_net_HEU, self.repaired_im_HEU, self.repaired_fm_HEU = pm4py.discover_petri_net_heuristics(self.repaired_log_df)

        params_discounted = helpers.get_params_discounted_from_pn(self.original_net)

        alignments_kpi_based = self.create_alignments_from_repaired_log(self.repaired_log_df, params_discounted)
        from pm4py.objects.petri_net.utils.align_utils import pretty_print_alignments
        #pretty_print_alignments(alignments_kpi_based)

        alignments_kpi_based_loc = self.create_alignments_from_repaired_log(self.repaired_log_df_loc, params_discounted)

        log_move_locations_per_trace = subprocess.get_log_move_locations_per_trace(net=self.original_net, alignments=alignments_kpi_based, initial_marking=self.original_im)
        self.kpi_based_pn, self.kpi_based_im, self.kpi_based_fm, self.kpi_based_n_changes = fahland.main(log_move_locations_per_trace=log_move_locations_per_trace,
                                                     original_pn=self.original_net, new_log=self.repaired_log_df)

        self.avoid_flower_pn, self.avoid_flower_im, self.avoid_flower_fm, self.avoid_flower_n_changes = fahland.main(log_move_locations_per_trace=log_move_locations_per_trace,
                                                     original_pn=self.original_net, new_log=self.repaired_log_df, avoid_flower=True)

        # consider moves with location as feature
        log_move_locations_per_trace_loc = subprocess.get_log_move_locations_per_trace(net=self.original_net, alignments=alignments_kpi_based_loc, initial_marking=self.original_im)
        self.loc_pn, self.loc_im, self.loc_fm, self.loc_n_changes = fahland.main(log_move_locations_per_trace=log_move_locations_per_trace_loc,
                                                     original_pn=self.original_net, new_log=self.repaired_log_df_loc)

        # petri net with FULL traces, not only those with good KPI values
        alignments_repair_all_traces = alignments_algo.apply(self.X_train_log,
                                                    self.original_net, self.original_im, self.original_fm,
                                                    parameters=params_discounted, variant=alignments_variants.dijkstra_less_memory)[0]
        log_move_locations_per_trace_full = subprocess.get_log_move_locations_per_trace(net=self.original_net,
                                                                                   alignments=alignments_repair_all_traces,
                                                                                   initial_marking=self.original_im)
        self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm, self.repair_all_traces_n_changes = fahland.main(log_move_locations_per_trace=log_move_locations_per_trace_full,
                                                                                        original_pn=self.original_net, new_log=self.X_train_log)        

    def print_conformant_kpi_values(self):
        results_columns = ['original', 'kpi_based', 'repair_all_traces',
                           'move_loc', 'avoid_flower',
                           'repaired_HEU', 'repaired_IM'#, 'repair_all_traces_train_cases'
                           ]
        self.X_test_cases_df['kpi'] = self.X_test_cases_df['case:concept:name'].map(self.target_KPI_values_per_case)
        results_df_columns_to_start_with = ['# fitting align-based', '# fitting token-based', 'mean KPI value', 'Median KPI value', 'Standard deviation of KPI v']
        results_df = pd.DataFrame(columns=results_df_columns_to_start_with)
        for before_or_repaired in results_columns:
            dict = {}
            for column, i in zip(results_df.columns, range(len(results_df.columns))):
                dict[column] = self.get_conformant_kpi_values(before_or_repaired)[i]
            results_df = results_df.append(dict, ignore_index=True)
        results_df = self.add_model_quality_attributes(results_df)
        results_df.index = results_columns
        display(results_df)

        print_further_details_for_masterarbeit_auswertung = False

        return_dict = {}
        if print_further_details_for_masterarbeit_auswertung:
            for x in results_df_columns_to_start_with[0:4]:
                return_dict[f"move-loc besser {x} als kpi-based"] = 0
                if results_df.loc["move_loc", x] > results_df.loc["kpi_based", x]:
                    return_dict[f"move-loc besser {x} als kpi-based"] = 1

            for x in results_df_columns_to_start_with[2:4]:
                for model_type in ['kpi_based', 'move_loc']:
                    return_dict[f"{model_type} besser {x} als repair_all_traces"] = 0
                    if results_df.loc[model_type, x] > results_df.loc['repair_all_traces', x]:
                        return_dict[f"{model_type} besser {x} als repair_all_traces"] = 1

            return_dict["avoid_flower praeziser als kpi-based"] = 0
            if results_df.loc["avoid_flower", 'precision'] > results_df.loc["kpi_based", 'precision']:
                return_dict["avoid_flower praeziser als kpi-based"] = 1

            return_dict["move-loc has more REPAIR STEPS than kpi-based"] = 0
            if self.loc_n_changes > self.kpi_based_n_changes:
                return_dict["move-loc has more REPAIR STEPS than kpi-based"] = 1

            return_dict["avoid-flower fits same # traces as kpi-based"] = 0
            if results_df.loc['avoid_flower', '# fitting align-based'] == results_df.loc["kpi_based", '# fitting align-based']:
                return_dict["avoid-flower fits same # traces as kpi-based"] = 1
        
        if self.run_in_ipynb:
            self.print_statistics()
        return return_dict

    def get_conformant_kpi_values(self, before_or_repaired):
        conformant = self.X_test_cases_df[self.X_test_cases_df[f'conformant_{before_or_repaired}']]
        kpi_values = list(conformant['kpi'])
        conformant_alignments = self.X_test_cases_df[self.X_test_cases_df[f'conformant_alignments_{before_or_repaired}']]
        kpi_values_align = list(conformant_alignments['kpi'])
        return [len(kpi_values_align), len(kpi_values), np.mean(kpi_values_align), np.median(kpi_values_align), np.std(kpi_values_align)]

    def add_model_quality_attributes(self, df):
        original_pt = pm4py.convert_to_process_tree(self.original_net, self.original_im, self.original_fm)
        original_zss_tree = tree_edit_distance.pt_to_zss_tree(original_pt)

        df['similarity'] = pd.Series([tree_edit_distance.editdist_zss_tree_petri(original_zss_tree, self.original_net, self.original_im, self.original_fm),
                                      self.kpi_based_n_changes,
                                      self.repair_all_traces_n_changes,
                                      self.loc_n_changes,
                                      self.avoid_flower_n_changes,
                                      '---',
                                      '---',
                                      #'---'
                                      ])
        
        df['simplicity'] = pd.Series([simplicity.apply(self.original_net),
                                      simplicity.apply(self.kpi_based_pn),
                                      simplicity.apply(self.repair_all_traces_pn),
                                      simplicity.apply(self.loc_pn),
                                      simplicity.apply(self.avoid_flower_pn),
                                      simplicity.apply(self.repaired_net_HEU),
                                      simplicity.apply(self.repaired_net_IM),
                                      #'See above'
                                      ])
        for quality in [precision, generalization]:
            quality_name = quality.__name__.split('.')[3]
            if quality == precision:
                variant = prec_variants.etconformance_token
            elif quality == generalization:
                variant = gen_variants.token_based
            df[quality_name] = pd.Series([quality.apply(self.X_test_log, self.original_net, self.original_im, self.original_fm, variant=variant),
                                          quality.apply(self.X_test_log, self.kpi_based_pn, self.kpi_based_im, self.kpi_based_fm, variant=variant),
                                          quality.apply(self.X_test_log, self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm, variant=variant),
                                          quality.apply(self.X_test_log, self.loc_pn, self.loc_im, self.loc_fm, variant=variant),
                                          quality.apply(self.X_test_log, self.avoid_flower_pn, self.avoid_flower_im, self.avoid_flower_fm, variant=variant),
                                          quality.apply(self.X_test_log, self.repaired_net_HEU, self.repaired_im_HEU, self.repaired_fm_HEU, variant=variant),
                                          quality.apply(self.X_test_log, self.repaired_net_IM, self.repaired_im_IM, self.repaired_fm_IM, variant=variant),                                      
                                          #quality.apply(self.X_train_log, self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm, variant=variant)
                                        ])
            
        df = self.add_replay_fitness(df)
        return df
    
    def add_replay_fitness(self, df: pd.DataFrame) -> pd.DataFrame:
        for fitness_type in ['log_fitness']: # 'percentage_of_fitting_traces', 'average_trace_fitness',
            replay_variant = replay_fitness.Variants.TOKEN_BASED
            df[fitness_type] = pd.Series([replay_fitness.apply(self.X_test_log, self.original_net, self.original_im, self.original_fm, variant=replay_variant)[fitness_type],
                                          replay_fitness.apply(self.X_test_log, self.kpi_based_pn, self.kpi_based_im, self.kpi_based_fm, variant=replay_variant)[fitness_type],
                                          replay_fitness.apply(self.X_test_log, self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm, variant=replay_variant)[fitness_type],
                                          replay_fitness.apply(self.X_test_log, self.loc_pn, self.loc_im, self.loc_fm, variant=replay_variant)[fitness_type],
                                          replay_fitness.apply(self.X_test_log, self.avoid_flower_pn, self.avoid_flower_im, self.avoid_flower_fm, variant=replay_variant)[fitness_type],
                                          replay_fitness.apply(self.X_test_log, self.repaired_net_HEU, self.repaired_im_HEU, self.repaired_fm_HEU, variant=replay_variant)[fitness_type],
                                          replay_fitness.apply(self.X_test_log, self.repaired_net_IM, self.repaired_im_IM, self.repaired_fm_IM, variant=replay_variant)[fitness_type]
                                          #replay_fitness.apply(self.X_train_log, self.repair_all_traces_pn, self.repair_all_traces_im, self.repair_all_traces_fm, variant=replay_variant)[fitness_type]
                                        ])
        return df

    
    def get_n_conformant_original_and_repaired(self, df):
        conf_df = df[df['conformant_original'] & df['conformant_repaired_IM']]
        return conf_df.shape[0]

    def get_n_conformant_kpi_based(self, df):
        conf_df = df[df['conformant_kpi_based']]
        return conf_df.shape[0]

    def print_statistics(self):
        stats_dict = {}
        stats_dict['n_train_cases'] = [len(self.X_train)]
        stats_dict['n_test_cases'] = [len(self.X_test)]
        stats_dict['n_conformant_kpi_based'] = [self.get_n_conformant_kpi_based(self.X_test_cases_df)]
        stats_df = pd.DataFrame(stats_dict)
        display(stats_df)
