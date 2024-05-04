# Process Model Repair

This application repairs Petri Net Models by adding behavior in order to improve a KPI of the process.

**Usage:**
* run.py [-h] --input INPUT_FILE_PATH [--out OUTPUT_PATH] [--matnr MATNR [MATNR ...]]
              [--target_value_col TARGET_VALUE_COL] [-lower_kpi_is_better]

**Input:** <br>
The INPUT_FILE_PATH must be the path to either a CSV or a XES file.
If it is a CSV-file it requires the columns:
* 'case:concept:name'
* 'concept:name'
* 'time:timestamp'
* 'material_id'
* 'Quality'
* 'Performance'
* 'Availability'

If it is a XES-file, case durations are required. These can be calculated and saved in a separate file in the Jupyter Notebook: 'create_case_durations.ipynb'.
The XES requires the fields
* 'case:concept:name'
* 'concept:name'
* 'time:timestamp'

**Output:** <br>
The application saves images to an output directory that can be determined using the flag '--out'. If it is not determined, a new directory named 'output' with a sub-directory named like the input-file is created and the output is saved there.
