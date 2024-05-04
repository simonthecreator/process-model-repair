import pm4py
import pandas as pd
import importer
from helpers import sollmodell_helpers, warnings_off, args_parser, data_prep
from main_repair import MainRepair
import math
import sys
import os

warnings_off.turn_off_warnings()

try:
   args = args_parser.get_arguments()
   print(f"Input Path is: {args.input_file_path}")
   print(f"lower_kpi_is_better: {args.lower_kpi_is_better}")

   input_file_name = os.path.split(args.input_file_path)[1].split('.')[0]

except ValueError as e:
   sys.exit(repr(e))

log = importer.read_from_input_file(args.input_file_path)

if "material_id" in log.columns:

   log['material_id'] = log['material_id'].apply(lambda x: str(int(x)) if type(x)==float and not math.isnan(x) else str(x) if type(x)==int else x)

   if args.matnr:
      matnrs = args.matnr
   else:
      grouped_by_matnr = log.groupby('material_id').agg({'case:concept:name': 'nunique', 'concept:name': 'nunique'}).sort_values(by='case:concept:name', ascending=False)
      top10_df = grouped_by_matnr.head(10)
      print(f"{top10_df}")
      top10_matnr = top10_df.reset_index()['material_id']

      matnrs = [ma if type(ma)==str else int(ma) for ma in top10_matnr]

else:
   matnrs = ["No matnrs in log"]

results_dict = {}
for matnr in matnrs:
   print("\n----------------------------\n")
   print(f"Material ID: {matnr}")
   print()

   specified_output_path = os.path.join(args.output_path, input_file_name)

   if matnr == "No matnrs in log":

      log_selected = log
      durations = importer.read_durations(file_path=args.input_file_path.__str__())

      kpi_dict = dict(zip(durations['case:concept:name'], durations.case_durations))

      soll_m, soll_im, soll_fm = sollmodell_helpers.create_soll_modell_by_variants(log=log_selected)

      thresholds_dict = {
         'mean': durations['case_durations'].mean(),
         'median': durations['case_durations'].median(),
         'q25': durations['case_durations'].quantile(q=0.25),
         'q75': durations['case_durations'].quantile(q=0.75),
         'q10': durations['case_durations'].quantile(q=0.10),
         'q05': durations['case_durations'].quantile(q=0.05),
         'q01': durations['case_durations'].quantile(q=0.01)
      }

      for name, value in thresholds_dict.items():
         n_cases_less_than_threshold = durations[durations['case_durations']<=value].shape[0]
         print(f"{name}-value: {value} with {n_cases_less_than_threshold} cases less than {name}-threshold")

      satisfactory_threshold = thresholds_dict['median']
      print(f"Dataset-shape <= Threshold: {durations[durations['case_durations']<=satisfactory_threshold].shape}")

   else:

      specified_output_path = os.path.join(specified_output_path, matnr)

      log_selected = log[log['material_id']==matnr]
      assert log_selected.shape[0] > 0, "Material ID not in data."

      # alternative Soll-Modell
      log_matnr_for_soll_m = log_selected.__deepcopy__()
      log_matnr_for_soll_m = log_matnr_for_soll_m[~log_matnr_for_soll_m['time:timestamp'].isna()]

      soll_m, soll_im, soll_fm = pm4py.discover_petri_net_inductive(log_matnr_for_soll_m,
                                                                     noise_threshold=0.8)
   
      log_selected = data_prep.remove_outliers_in_kpi_values(log_selected, outliers_threshold = 5)

      if args.target_value_col == 'OEE':

         log_selected[args.target_value_col] = log_selected['Quality'] * log_selected['Performance'] * log_selected['Availability']

         grouped_for_kpi = log_selected.groupby('case:concept:name').agg({'concept:name': 'count', args.target_value_col: 'mean',
                                                         'Quality': 'mean', 'Performance': 'mean', 'Availability': 'mean'}).reset_index()

         target_cols = [args.target_value_col, 'Quality', 'Performance', 'Availability']

         log_selected = log_selected.merge(grouped_for_kpi[['case:concept:name']+target_cols], on='case:concept:name', suffixes=['_activity', ''])

         kpi_dict = dict(zip(grouped_for_kpi['case:concept:name'], grouped_for_kpi[args.target_value_col]))

         # set threshold
         satisfactory_threshold = grouped_for_kpi[args.target_value_col].quantile(q=0.25)
         print(f"Dataset-shape >= Threshold: {grouped_for_kpi[grouped_for_kpi[args.target_value_col]>=satisfactory_threshold].shape}")

         log_selected = log_selected[log_selected['time:timestamp'].notna()]
   
   print(f"satisfactory_threshold: {satisfactory_threshold}")

   # create folder for output files
   if not os.path.exists(specified_output_path):
      os.makedirs(specified_output_path)

   # set up MainRepair object
   repairer = MainRepair(log_selected,
                         soll_m,
                         soll_im,
                         soll_fm,
                         target_KPI_values_per_case = kpi_dict,
                         satisfactory_values=[satisfactory_threshold],
                         output_dir = specified_output_path,
                         lower_KPI_is_better=args.lower_kpi_is_better,
                         run_in_ipynb = False)

   try:    
      repairer.main()
      any_repaired_version_better_than_repair_all = repairer.print_conformant_kpi_values()
      if any_repaired_version_better_than_repair_all:
         results_dict[matnr] = any_repaired_version_better_than_repair_all

   except AssertionError as ae:
      print(f"AssertionError occured: {ae}")
      continue

if results_dict:
   results_dict_to_df = pd.DataFrame.from_dict({j: results_dict[j] 
                              for j in results_dict.keys()},
                        orient='columns')
   print(results_dict_to_df)