import argparse
import sys
from extract_database import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse to query MIMIC/eICU data")
    parser.add_argument("--database", type=str, default='MIMIC', choices=['MIMIC', 'eICU'])
    parser.add_argument("--project_id", type=str, default='lucid-inquiry-337016',
                        help='Specify the Bigquery billing project')
    parser.add_argument("--age_min", type=int, default=18, help='Min patient age to query')
    parser.add_argument("--los_min", type=int, default=24, help='Min ICU LOS in hour')
    parser.add_argument("--los_max", type=int, default=240, help='Max ICU LOS in hour')
    parser.add_argument("--patient_group", type=str, default='Generic', choices=['Generic', 'sepsis_3', 'ARF', 'shock', 'COPD', 'CHF'],
                        help='Specific groups to extract')
    parser.add_argument("--custom_id", action='store_true', default=False, help="Whether use custom stay ids")
    parser.add_argument('--customid_dir', required='--custom_id' in sys.argv, help="Specify custom id dir")
    parser.add_argument("--exit_point", type=str, default='All', choices=['All', 'Raw', 'Outlier_removal', 'Impute'],
                        help='Where to stop the pipeline')
    parser.add_argument("--no_removal", action='store_true', default=False, help="When set to True, no outlier removal")
    parser.add_argument("--norm_eicu", type=str, default='MIMIC', choices=['MIMIC', 'eICU'],
                        help="Whether use MIMIC mean and std to standardize eICU variables")
    parser.add_argument("--time_window", type=int, default=1, help='Time window to aggregate the data')
    parser.add_argument("--output_dir", type=str, default='./output')
    args = parser.parse_args()
    if args.database == 'MIMIC':
        extract_mimic(args)
    elif args.database == 'eICU':
        extract_eicu(args)

