import json
import os
from pathlib import Path
import numpy as np
from opts import parse_opts
from get_radiomics import get_radiomics
from svc import select_feature, train_and_predict_svm, train_and_test_svm

# PROJECT_PATH = '/Users/ranyan/workspace/Project_nac_pcr_pre'
PROJECT_PATH = '/media/hdd1/ran/Project_nac_pcr_pre'

def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)

def get_opt():
    opt = parse_opts()
    
    print(opt)
    with open(os.path.join(PROJECT_PATH, 'results', 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file, default=json_serial)

    return opt


if __name__ == '__main__':

    opt = get_opt()


    # # get radiomics for training data
    # get_radiomics(opt, is_training = True)

    # # get radiomics for testing data
    # get_radiomics(opt, is_training = False)

    # svm training using cross validation
    result = []
    dtype = [('c', float), ('selected_feature_number', int), ('mean_auc', float), ('lower_ci_auc', float), ('upper_ci_auc', float), ('mean_train_auc', float)]
    for c in opt.svc_c_list:
        for selected_feature_number in opt.selected_feature_number_list:
            mean_auc, lower_ci_auc, upper_ci_auc, mean_train_auc = train_and_predict_svm(c, selected_feature_number, opt)
            with open(os.path.join(PROJECT_PATH, 'results', 'output.txt'), 'a') as output_file:
                print('c:'+ str(c) + ', Feature_number' + str(selected_feature_number) + ', AUC[CI]:' + str(format(mean_auc, '.3f'))+ '[' + str(format(lower_ci_auc, '.3f')) + '-' + str(format(upper_ci_auc, '.3f')) + '], train_auc:' + str(format(mean_train_auc, '.3f')), file=output_file)
            result.append((c, selected_feature_number, mean_auc, lower_ci_auc, upper_ci_auc, mean_train_auc))
    result = np.array(result, dtype=dtype)

    c_best, selected_feature_number_best, mean_auc_best, lower_ci_auc, upper_ci_auc, train_auc_best = np.sort(result, order='mean_auc')[0]
    with open(os.path.join(PROJECT_PATH, 'results', 'output.txt'), 'a') as output_file:
        print('Best c:' + str(c_best) + ', Best feature number:' + str(selected_feature_number_best) + ', Max AUC[CI]:'+ str(format(mean_auc_best, '.3f'))+ '[' + str(format(lower_ci_auc, '.3f')) + '-' + str(format(upper_ci_auc, '.3f')) + '], train_auc:' + str(format(train_auc_best, '.3f')), file=output_file)

    # svm training using all training data and testing using test dataset
    model_path = os.path.join(PROJECT_PATH, 'results', 'model_' + opt.radiomics_parameter_name 
                                + '_features' + str(selected_feature_number_best) + '_c' + str(c_best) + '.p')
    predict_result_path = os.path.join(PROJECT_PATH, 'results', 'predictResults_' + opt.radiomics_parameter_name 
                                + '_features' + str(selected_feature_number_best) + '_c' + str(c_best) + '.csv')

    auc, selected_feature_name = train_and_test_svm(opt, selected_feature_number_best, c_best, model_path, predict_result_path)
    with open(os.path.join(PROJECT_PATH, 'results', 'output.txt'), 'a') as output_file:
        print('Test AUC:'+ str(format(auc, '.3f')) + ', feature names:' + str(selected_feature_name), file=output_file)
