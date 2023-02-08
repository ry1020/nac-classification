import argparse 
from pathlib import Path
import os

# PROJECT_PATH = '/Users/ranyan/workspace/Project_nac_pcr_pre'
PROJECT_PATH = '/media/hdd1/ran/Project_nac_pcr_pre'

# Training settings
def parse_opts():
    parser = argparse.ArgumentParser(description='NAC Classification Project')
    parser.add_argument('--is_training',
                        action='store_true',
                        help='If true, training is performed, otherwise testing is performed')                    
    parser.add_argument('--image_name_keyword',
                        default='DCE-',
                        type=str,
                        help='Choose from (DCE- | SER- | PE2- | PE56-)')
    parser.add_argument('--dce_phase_select',
                        default=0,
                        type=int,
                        help='Choose from (0, 2)')
    parser.add_argument('--scan_time_keyword',
                        default='T1-',
                        type=str,
                        help='Choose from (T0- | T1- | T2-)')
    parser.add_argument('--mask_name_keyword',
                        default='Analysis Mask',
                        type=str,
                        help='Choose from (Analysis Mask)')
    parser.add_argument('--start_patient',
                        default=0,
                        type=int,
                        help='start from which patient number')
    parser.add_argument('--is_plot',
                        action='store_true',
                        help='If true, the image will be plotted.')

    parser.add_argument('--selected_features_number',
                        default=10,
                        type=int,
                        help='Number of features to input in SVM')
    parser.add_argument('--radiomics_parameters_name',
                        default='ParamSetting3_2dFalse_normFalse_binWidth15',
                        type=str,
                        help='Choose 2d, normalization, binWidth/binCount')
    parser.add_argument('--cross_validation_fold_number',
                        default=5,
                        type=int,
                        help='Number of cross validation fold')
    args = parser.parse_args()

    return args
    