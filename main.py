import json
import os
from pathlib import Path
from multiprocessing.pool import ThreadPool

from opts import parse_opts
from get_radiomics import get_radiomics
from svc import select_features, train_and_predict_svm, train_svm_all_data, test_svm

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

    # select features
    data_x, data_y, final_features_names = select_features(opt)

    # svm training using cross validation
    pool = ThreadPool(len(opt.svc_c_list))
    threads = []
    for c in opt.svc_c_list:
        threads.append(pool.apply_async(train_and_predict_svm, args=(data_x, data_y, c, opt.svc_kernel, opt.cross_validation_fold_number)))
    pool.close()
    pool.join()
    results = [p.get() for p in threads]
    print('c     auc   acc')
    for auc, acc, c in results:
        print(format(c, '.3f'), format(auc, '.3f'), format(acc, '.3f'))
    max_auc, max_acc, best_c = sorted(results, reverse=True)[0]
    print('Best c:' + str(best_c) + ',max auc:'+ str(max_auc) + ',max acc:' + str(max_acc))

    model_path = os.path.join(PROJECT_PATH, 'results', 'model_' + opt.radiomics_parameters_name 
                                + '_features' + str(opt.selected_features_number) + '_c' + str(best_c) + '.p')
    predict_result_path = os.path.join(PROJECT_PATH, 'results', 'predictResults_' + opt.radiomics_parameters_name 
                                + '_features' + str(opt.selected_features_number) + '_c' + str(best_c) + '.csv')
    
    # svm training using all training data
    training_auc, training_acc = train_svm_all_data(opt, final_features_names, best_c, model_path)
    print('training auc:' + str(format(training_auc, '.3f')) + ', acc:', format(training_acc, '.3f'))
    
    # svm testing
    testing_auc, testing_acc = test_svm(opt, final_features_names, model_path, predict_result_path)
    print('testing auc:' + str(format(testing_auc, '.3f')) + ', acc:', format(testing_acc, '.3f'))