from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
import numpy as np
import pickle
import pandas as pd
import os
import datetime
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
# from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._testing import ignore_warnings
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# PROJECT_PATH = '/Users/ranyan/workspace/Project_nac_pcr_pre'
PROJECT_PATH = '/media/hdd1/ran/Project_nac_pcr_pre'

FEATURE_NAME_KEY_WORDS = ['firstorder', 'original_shape', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
TRAINING_CLINICAL_INFO_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'BMMR2_documents', 'BMMR2_Training_set_cases_with_clinical_data_20210520.csv')
TESTING_CLINICAL_INFO_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'BMMR2_documents', 'BMMR2_Testing_set_cases_with_clinical_data_20210520.csv')

# t0的min可能为零
DELETE_FEATURES = ['firstorder_Minimum']

T_SUFFIX = ['t0', 't2']

def get_feature_folder_path(radiomics_parameter_name, is_training):
    if is_training:
        feature_folder_path = os.path.join(PROJECT_PATH, 'data', 'mydata', radiomics_parameter_name, 'training_data')
    else:
        feature_folder_path = os.path.join(PROJECT_PATH, 'data', 'mydata', radiomics_parameter_name, 'testing_data')
    return feature_folder_path

def get_feature_prefix_category_set(feature_folder_path):
    feature_prefix_category_set = list(set([category[:-2] for category in os.listdir(feature_folder_path) if 'DS_Store' not in category]))
    feature_prefix_category_set = [ category for category in feature_prefix_category_set if 'feature' in category ]  # remove 'img' and 'mask' files
    return feature_prefix_category_set

def get_clinical_data(clinical_data_path):
    csv_data = pd.read_csv(clinical_data_path)
    name = csv_data.columns
    for col in range(5, 9):    # Change 'race','Ltype', 'hrher4g', 'SBRgrade' into dummy feature
        prefix = name[col]
        dummy_feature = pd.get_dummies(csv_data[prefix], prefix=prefix)
        csv_data = pd.concat([csv_data, dummy_feature], axis=1)
        csv_data.drop([prefix], axis=1, inplace=True)

    patient_id_list = csv_data[name[1]].to_list()
    labels = [1 if label == 'pCR' else 0 for label in csv_data[name[10]].to_list()]

    csv_data.drop([name[0], name[1], name[2], name[3], name[10], name[11], 'race_Unknown'],
                  axis=1, inplace=True)   # remove 'Patient ID Number', 'Patient ID DICOM', 'analy', 'elig', 'pcr', 'Split'

    return csv_data, patient_id_list, labels

def is_feature(feature_key):
    if '_ser_' in feature_key and ('_glcm_' in feature_key or '_glrlm_' in feature_key
                                   or '_glszm_' in feature_key or '_ngtdm_' in feature_key or '_gldm_' in feature_key):
        return False
    for word in DELETE_FEATURES:
        if word in feature_key:
            return False
    for word in FEATURE_NAME_KEY_WORDS:
        if word in feature_key:
            return True
    return False

def is_both_t0_t2_exist(patient_id, feature_folder_path):
    feature_prefix_category_set = get_feature_prefix_category_set(feature_folder_path)
    for category in feature_prefix_category_set:
        for t in T_SUFFIX:
            path = os.path.join(feature_folder_path, category + t, patient_id + '.p')
            if not os.path.exists(path):
                return False
    return True

def get_all_feature_data_for_single_patient(patient_id, feature_folder_path):
    feature_name = []
    feature_values = []
    feature_prefix_category_set = get_feature_prefix_category_set(feature_folder_path)

    for feature_prefix in feature_prefix_category_set:
        t0_feature_file = os.path.join(feature_folder_path, feature_prefix + 't0', patient_id + '.p')
        t2_feature_file = os.path.join(feature_folder_path, feature_prefix + 't2', patient_id + '.p')
        t0_feature = pickle.load(open(t0_feature_file, 'rb'))
        t2_feature = pickle.load(open(t2_feature_file, 'rb'))

        for t0_key, val in t0_feature.items():
            if not is_feature(t0_key):
                continue

            key_prefix = t0_key[:-2]
            t2_key = key_prefix + 't2'
            t2_val = float(t2_feature[t2_key])
            t0_val = float(val)

            # Since it represents asymmetry, could be positive or negative, so get absolute value
            if 'firstorder_Skewness' in t0_key or 'ClusterShade' in t0_key:
                t0_val = abs(t0_val)
                t2_val = abs(t2_val)

            feature_name.append(t0_key)
            feature_name.append(t2_key)
            feature_name.append(key_prefix + 'delta')
            feature_values.append(t0_val)
            feature_values.append(t2_val)
            feature_values.append((t2_val - t0_val) / t0_val)

    return feature_name[:], feature_values[:]

def combine_data(radiomics_parameter_name, is_training):
    feature_folder_path = get_feature_folder_path(radiomics_parameter_name, is_training)

    clinical_info_file_path = TRAINING_CLINICAL_INFO_FILE_PATH if is_training else TESTING_CLINICAL_INFO_FILE_PATH
    clinical_data, patient_id_list, labels = get_clinical_data(clinical_info_file_path)
    clinical_feature_name = clinical_data.columns.tolist()
    patient_id_list = [patient_id for patient_id in patient_id_list if is_both_t0_t2_exist(patient_id, feature_folder_path)] # 因为算t0和算t2的病人有不一样的
    number_of_data = len(patient_id_list)
    number_of_clinical_feature = len(clinical_feature_name)

    clinical_data = clinical_data.to_numpy()

    data_x = []
    data_y = []
    all_feature_name = []

    for i in range(number_of_data):
        patient_id = patient_id_list[i]
        feature_name, feature_values = get_all_feature_data_for_single_patient(patient_id, feature_folder_path)

        for j in range(number_of_clinical_feature):
            feature_name.append(clinical_feature_name[j])
            feature_values.append(clinical_data[i][j])

        all_feature_name, feature_values = zip(*sorted(zip(feature_name, feature_values)))
        data_x.append(feature_values)
        data_y.append(labels[i])

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    scaler = StandardScaler()
    scaler.fit(data_x)
    data_x = scaler.transform(data_x)

    return data_x, data_y, all_feature_name, patient_id_list

def calculate_vif(data_x, feature_name):
    vif = pd.DataFrame()
    vif['index'] = feature_name
    vif['VIF'] = [variance_inflation_factor(data_x,i) for i in range(data_x.shape[1])]
    return vif

def select_feature(opt, selected_feature_number, data_x, data_y, feature_name):
    lasso = Lasso(alpha = opt.lasso_alpha, tol=opt.lasso_tolerance).fit(data_x, data_y)
    importance = np.abs(lasso.coef_)
    sorted_importance = sorted(importance)
    threshold = sorted_importance[-opt.lasso_feature_number]

    lasso_mask = np.array([v > threshold for v in importance])
    selected_data_x = data_x[:, lasso_mask]
    selected_feature_name = np.array(feature_name)[lasso_mask]

    vif = calculate_vif(selected_data_x, selected_feature_name)
    while (vif['VIF'] > 5).any():       # VIF threshold can be changed
        remove = vif.sort_values(by='VIF',ascending=False)['index'][:1].values[0]
        index = np.where(selected_feature_name == remove)[0]
        selected_feature_name = np.delete(selected_feature_name, index)
        selected_data_x = np.delete(selected_data_x, index, axis=1)
        vif = calculate_vif(selected_data_x, selected_feature_name)


    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.plot(range(334), importance)
    # ax2.imshow(corr_m, aspect = 'auto')
    # plt.show()

    lasso_cv = LassoCV(tol=opt.lassoCV_tolerance, n_jobs=-1).fit(selected_data_x, data_y)
    sfs_forward = SequentialFeatureSelector(lasso_cv, 
                                            n_features_to_select=selected_feature_number, 
                                            direction="forward", n_jobs=-1).fit(selected_data_x, data_y)

    sfs_mask = sfs_forward.get_support()

    selected_data_x = selected_data_x[:, sfs_mask]
    selected_feature_name = np.array(selected_feature_name)[sfs_mask]

    df = pd.DataFrame(selected_data_x)
    corr_m = df.corr().abs()
    corr_m = np.array(corr_m).tolist()

    cor_mask = []
    for i in range(selected_feature_number):
        cor_mask.append(all([corr_m[i][j] <= opt.feature_correlation_threshold for j in range(i+1, selected_feature_number)]))

    cor_mask = np.array(cor_mask)
    selected_data_x = selected_data_x[:, cor_mask]
    selected_feature_name = selected_feature_name[cor_mask]

    return selected_data_x, data_y, selected_feature_name.tolist()

def train_and_predict_svm(c, selected_feature_number, opt):
    auc = []
    train_auc = []
    # acc = []
    # train_acc = []

    data_x, data_y, feature_name, _ = combine_data(opt.radiomics_parameter_name, is_training = True)

    with open(opt.result_txt_path, 'a') as output_file:
        print('Cross validation', file=output_file)
    
    # This cross-validation object is a variation of KFold that returns stratified folds. 
    # The folds are made by preserving the percentage of samples for each class.
    # Stratified sampling
    for train_index, val_index in StratifiedKFold(n_splits = opt.cross_validation_fold_number).split(data_x, data_y):
        x_train, x_val = data_x[train_index], data_x[val_index]
        y_train, y_val = data_y[train_index], data_y[val_index]

        selected_x_train, y_train, selected_feature_name = select_feature(opt, selected_feature_number, x_train, y_train, feature_name)
        with open(opt.result_txt_path, 'a') as output_file:
            print('c:' + str(c) + ', feature number:' + str(selected_feature_number) + ', Selected features:' + str(selected_feature_name), file=output_file)

        masks = np.array([name in selected_feature_name for name in feature_name])
        selected_x_val = np.array(x_val)[:, masks]

        model = SVC(kernel=opt.svc_kernel, C=c).fit(selected_x_train, y_train)

        y_decision = model.decision_function(selected_x_val)
        auc.append(roc_auc_score(y_val, y_decision))
        train_auc.append(roc_auc_score(y_train, model.decision_function(selected_x_train)))
        
        # y_predict = model.predict(selected_x_val)
        # acc.append(accuracy_score(y_val, y_predict))
        # train_acc.append(accuracy_score(y_train, model.predict(selected_x_train)))

    mean_auc = np.mean(np.array(auc))
    std_auc = np.std(np.array(auc))

    # Calculate the 95% confidence interval for the AUC scores
    n = len(auc)
    t_value = 2.776 # t-value for a 95% confidence interval with n-1 degrees of freedom (n = 5 fold)
    lower_ci_auc = mean_auc - t_value * (std_auc / np.sqrt(n))
    upper_ci_auc = mean_auc + t_value * (std_auc / np.sqrt(n))

    mean_train_auc = np.mean(np.array(train_auc))

    return mean_auc, lower_ci_auc, upper_ci_auc, mean_train_auc

def train_and_test_svm(opt, selected_feature_number, c, model_path, predict_result_path):
    data_x, data_y, feature_name, _ = combine_data(opt.radiomics_parameter_name, is_training = True)
    
    selected_data_x, data_y, selected_feature_name = select_feature(opt, selected_feature_number, data_x, data_y, feature_name)

    with open(opt.result_txt_path, 'a') as output_file:
        print('Final train', file=output_file)
        print('c:' + str(c) + ', feature number:' + str(selected_feature_number) + ', Selected features:' + str(selected_feature_name), file=output_file)

    model = SVC(kernel=opt.svc_kernel, C=c).fit(selected_data_x, data_y)

    pickle.dump(model, open(model_path, 'wb'))

    clinical_info_file_path = TESTING_CLINICAL_INFO_FILE_PATH
    _, patient_id_list, _ = get_clinical_data(clinical_info_file_path)
    data_x, data_y, feature_name, _ = combine_data(opt.radiomics_parameter_name, is_training = False)

    masks = np.array([name in selected_feature_name for name in feature_name])
    selected_data_x = data_x[:, masks]

    y_predict = model.predict(selected_data_x)
    y_decision = model.decision_function(selected_data_x)
    auc = roc_auc_score(data_y, y_decision)
    # acc = accuracy_score(data_y, y_predict)

    y_predict = y_predict.tolist()
    index = patient_id_list.index('ACRIN-6698-641246')  # 该patient因为没有同时具有t0和t2的feature所以被去除了，默认是'pCR'
    y_predict = y_predict[0:index] + [1] + y_predict[index:]

    data = {'patient_id': patient_id_list, 'pCR/non-pCR': y_predict}
    pd.DataFrame(data).to_csv(predict_result_path)
    
    return auc, selected_feature_name

