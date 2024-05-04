import argparse
from dataclasses import dataclass
from enum import Enum
import pathlib
import os
import os
from typing import List

class KpiAttribute(Enum):
    """Enum for supported KPI value attributes"""
    OEE = 'OEE'
    ThroughputTime = 'throughput_time'

@dataclass
class Args:
    """Holds the arguments from program call"""
    input_file_path: str
    output_path: str
    matnr: List[str]
    target_value_col: KpiAttribute
    lower_kpi_is_better: bool

__parser = argparse.ArgumentParser()

__parser.add_argument('--input',
                      action='store',
                      help='path to input file',
                      dest='input_file_path',
                      type=pathlib.Path,
                      required=True)

__parser.add_argument('--out',
                      action='store',
                      help='''the output path specifying where the generated files will be stored.
                      If not specified, a subdirectory will be created in the input path.''',
                      dest='output_path',
                      type=pathlib.Path,)

__parser.add_argument('--matnr',
                      nargs='+',
                      action='store',
                      dest='matnr',
                      help='Specify the material numbers to be used.',
                      required=False,
                      default=[])

__parser.add_argument('--target_value_col',
                      dest='target_value_col',
                      help='Column which should be used as target value to be optimized.',
                      action='store',
                      default='OEE')

__parser.add_argument('-lower_kpi_is_better',
                      dest='lower_kpi_is_better',
                      help='Boolean that defines whether lower KPI value is more desirable',
                      action='store_true') # store_true means it is False if flag is not set

def get_arguments() -> Args:
    """Parse the arguments of the program call.

    Returns:
        Args: Args object.
    """
    parsed = __parser.parse_args()
    
    try:
        target_value_col = parsed.target_value_col
    except ValueError as e:
        raise ValueError(
            f'{str(e)}. Supported KPI Attributes: {[i.value for i in KpiAttribute]}')

    args = Args(input_file_path=parsed.input_file_path,
                output_path=parsed.output_path,
                matnr=parsed.matnr,
                target_value_col=target_value_col,
                lower_kpi_is_better=parsed.lower_kpi_is_better)

    # check output path
    args.output_path = args.output_path or os.path.join(os.path.dirname(args.input_file_path), 'output')

    return args