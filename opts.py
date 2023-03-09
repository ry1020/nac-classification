import argparse 
from pathlib import Path
import os

# PROJECT_PATH = '/Users/ranyan/workspace/Project_nac_pcr_pre'
PROJECT_PATH = '/media/hdd1/ran/Project_nac_pcr_pre'

# Training settings
def parse_opts():
    parser = argparse.ArgumentParser(description='NAC Classification Project')
    parser.add_argument('--result_txt_path',
                    default = os.path.join(PROJECT_PATH, 'results', 'output_vif.txt'),
                    type=Path,
                    help='Result txt path')
    parser.add_argument('--image_name_keyword',
                        default='DCE-',
                        type=str,
                        help='Choose from (DCE- | SER- | PE2- | PE56-)')
    parser.add_argument('--dce_phase_select',
                        default=0,
                        type=int,
                        help='Choose from (0, 2)')
    parser.add_argument('--scan_time_keyword',
                        default='T0-',
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

    parser.add_argument('--selected_feature_number_list',
                        default=[5, 10, 15],
                        type=int,
                        nargs='+',
                        help='Number of features to input in SVM [5, 10, 15]')
    parser.add_argument('--radiomics_parameter_name',
                        default='ParamSetting3_2dFalse_normFalse_binWidth15',
                        type=str,
                        help='Choose 2d, normalization, binWidth/binCount')
    parser.add_argument('--cross_validation_fold_number',
                        default=5,
                        type=int,
                        help='Number of cross validation fold')
    parser.add_argument('--lasso_alpha',
                        default=0.003,
                        type=float,
                        help='Constant that multiplies the L1 term, controlling regularization strength')
    parser.add_argument('--lasso_feature_number',
                        default=100,
                        type=int,
                        help='Number of features selected by lasso')
    parser.add_argument('--lasso_tolerance',
                        default=0.002,
                        type=float,
                        help='The tolerance for the optimization (0.02 | 0.002)')
    parser.add_argument('--lassoCV_tolerance',
                        default=0.002,
                        type=float,
                        help='The tolerance for the optimization (0.02 | 0.002)')
    parser.add_argument('--feature_correlation_threshold',
                        default=0.95,
                        type=float,
                        help='Feature will be omitted if correlation > threshold')
    parser.add_argument('--svc_kernel',
                        default='poly',
                        type=str,
                        help='Choose from (linear | poly | rbf | sigmoid)')
    parser.add_argument('--svc_c_list',
                        default= [0.03, 0.1, 0.3, 1, 3, 10, 30],
                        type=float,
                        nargs='+',
                        help='Regularization parameter. The strength of the regularization is' +
                              'inversely proportional to C. The penalty is a squared l2 penalty.' +
                              'Choose from [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]')
    args = parser.parse_args()

    return args
    