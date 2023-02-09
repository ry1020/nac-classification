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

    get_radiomics(opt)

    # select features
    data_x, data_y, final_features_names = select_features(opt)

    # svm training using cross validation
    c_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    pool = ThreadPool(len(c_list))
    threads = []
    for c in c_list:
        threads.append(pool.apply_async(train_and_predict_svm, args=(data_x, data_y, c, opt.cross_validation_fold_number)))
    pool.close()
    pool.join()
    results = [p.get() for p in threads]
    for auc, acc, c in results:
        print(c, auc, acc)
    max_auc, max_acc, best_c = sorted(results, reverse=True)[0]
    print('Best c {}, max auc {}, max acc {}'.format(best_c, max_auc, max_acc))

    model_path = os.path.join(PROJECT_PATH, 'results', 'model_{}_features{}_c{}.p'.format(opt.radiomics_parameters_name,str(opt.selected_features_number),str(best_c)))
    predict_result_path = os.path.join(PROJECT_PATH, 'results', 'predictResults_{}_features{}_c{}.csv'.format(opt.radiomics_parameters_name,str(opt.selected_features_number),str(best_c)))
    
    # svm training using all training data
    training_auc, training_acc = train_svm_all_data(opt, final_features_names, best_c, model_path)
    print('training auc, acc: ', training_auc, training_acc)
    
    # svm testing
    testing_auc, testing_acc = test_svm(opt, final_features_names, model_path, predict_result_path)
    print('training auc, acc: ', testing_auc, testing_acc)