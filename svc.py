from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
import numpy as np
import pickle
import pandas as pd
import os
import datetime
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import scale
# from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._testing import ignore_warnings
import matplotlib.pyplot as plt

# PROJECT_PATH = '/Users/ranyan/workspace/Project_nac_pcr_pre'
PROJECT_PATH = '/media/hdd1/ran/Project_nac_pcr_pre'

FEATURE_NAME_KEY_WORDS = ['firstorder', 'original_shape', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
TRAINING_CLINICAL_INFO_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'BMMR2_documents', 'BMMR2_Training_set_cases_with_clinical_data_20210520.csv')
TESTING_CLINICAL_INFO_FILE_PATH = os.path.join(PROJECT_PATH, 'data', 'BMMR2_documents', 'BMMR2_Testing_set_cases_with_clinical_data_20210520.csv')

# t0的min可能为零
DELETE_FEATURES = ['firstorder_Minimum']

T_SUFFIX = ['t0', 't2']

def get_features_folder_path(radiomics_parameters_name, is_training):
    if is_training:
        features_folder_path = os.path.join(PROJECT_PATH, 'data', 'mydata', radiomics_parameters_name, 'training_data')
    else:
        features_folder_path = os.path.join(PROJECT_PATH, 'data', 'mydata', radiomics_parameters_name, 'testing_data')
    return features_folder_path

def get_feature_prefix_category_set(features_folder_path):
    feature_prefix_category_set = list(set([category[:-2] for category in os.listdir(features_folder_path) if 'DS_Store' not in category]))
    feature_prefix_category_set = [ category for category in feature_prefix_category_set if 'features' in category ]  # remove 'img' and 'mask' files
    return feature_prefix_category_set

def get_clinical_data(clinical_data_path):
    csv_data = pd.read_csv(clinical_data_path)
    names = csv_data.columns
    for col in range(5, 9):    # Change 'race','Ltype', 'hrher4g', 'SBRgrade' into dummy features
        prefix = names[col]
        dummy_feature = pd.get_dummies(csv_data[prefix], prefix=prefix)
        csv_data = pd.concat([csv_data, dummy_feature], axis=1)
        csv_data.drop([prefix], axis=1, inplace=True)

    patient_id_list = csv_data[names[1]].to_list()
    labels = [1 if label == 'pCR' else 0 for label in csv_data[names[10]].to_list()]

    csv_data.drop([names[0], names[1], names[2], names[3], names[10], names[11], 'race_Unknown'],
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


def is_both_t0_t2_exist(patient_id, features_folder_path):
    feature_prefix_category_set = get_feature_prefix_category_set(features_folder_path)
    for category in feature_prefix_category_set:
        for t in T_SUFFIX:
            path = os.path.join(features_folder_path, category + t, patient_id + '.p')
            if not os.path.exists(path):
                return False
    return True


def get_all_feature_data_for_single_patient(patient_id, features_folder_path):
    feature_names = []
    feature_values = []
    feature_prefix_category_set = get_feature_prefix_category_set(features_folder_path)

    for feature_prefix in feature_prefix_category_set:
        t0_feature_file = os.path.join(features_folder_path, feature_prefix + 't0', patient_id + '.p')
        t2_feature_file = os.path.join(features_folder_path, feature_prefix + 't2', patient_id + '.p')
        t0_features = pickle.load(open(t0_feature_file, 'rb'))
        t2_features = pickle.load(open(t2_feature_file, 'rb'))

        for t0_key, val in t0_features.items():
            if not is_feature(t0_key):
                continue

            key_prefix = t0_key[:-2]
            t2_key = key_prefix + 't2'
            t2_val = float(t2_features[t2_key])
            t0_val = float(val)

            # Since it represents asymmetry, could be positive or negative, so get absolute value
            if 'firstorder_Skewness' in t0_key or 'ClusterShade' in t0_key:
                t0_val = abs(t0_val)
                t2_val = abs(t2_val)

            feature_names.append(t0_key)
            feature_names.append(t2_key)
            feature_names.append(key_prefix + 'delta')
            feature_values.append(t0_val)
            feature_values.append(t2_val)
            feature_values.append((t2_val - t0_val) / t0_val)

    return feature_names[:], feature_values[:]


def combine_data(radiomics_parameters_name, is_training):
    features_folder_path = get_features_folder_path(radiomics_parameters_name, is_training)

    clinical_info_file_path = TRAINING_CLINICAL_INFO_FILE_PATH if is_training else TESTING_CLINICAL_INFO_FILE_PATH
    clinical_data, patient_id_list, labels = get_clinical_data(clinical_info_file_path)
    clinical_feature_names = clinical_data.columns.tolist()
    patient_id_list = [patient_id for patient_id in patient_id_list if is_both_t0_t2_exist(patient_id, features_folder_path)] # 因为算t0和算t2的病人有不一样的
    number_of_data = len(patient_id_list)
    number_of_clinical_features = len(clinical_feature_names)

    clinical_data = clinical_data.to_numpy()

    data_x = []
    data_y = []
    all_feature_names = []

    for i in range(number_of_data):
        patient_id = patient_id_list[i]
        feature_names, feature_values = get_all_feature_data_for_single_patient(patient_id, features_folder_path)

        for j in range(number_of_clinical_features):
            feature_names.append(clinical_feature_names[j])
            feature_values.append(clinical_data[i][j])

        all_feature_names, feature_values = zip(*sorted(zip(feature_names, feature_values)))
        data_x.append(feature_values)
        data_y.append(labels[i])

    data_x = scale(np.array(data_x))
    data_y = np.array(data_y)

    return data_x, data_y, all_feature_names, patient_id_list

@ignore_warnings(category=ConvergenceWarning)
def select_features(opt):
    data_x, data_y, feature_names, _ = combine_data(opt.radiomics_parameters_name, is_training = True)

    lasso = Lasso(alpha = opt.lasso_alpha).fit(data_x, data_y)
    importance = np.abs(lasso.coef_)
    sorted_importance = sorted(importance)
    threshold = sorted_importance[-opt.lasso_feature_number]

    first_mask = np.array([v > threshold for v in importance])
    data_x = data_x[:, first_mask]
    feature_names = np.array(feature_names)[first_mask]

    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.plot(range(334), importance)
    # ax2.imshow(corr_m, aspect = 'auto')
    # plt.show()

    print('feature select starts at: ', datetime.datetime.now())
    lasso_cv = LassoCV(tol=opt.lassoCV_tolerance, n_jobs=-1).fit(data_x, data_y)
    sfs_forward = SequentialFeatureSelector(lasso_cv, 
                                            n_features_to_select=opt.selected_features_number, 
                                            direction="forward").fit(data_x, data_y)

    masks = sfs_forward.get_support()

    print('feature select ends at: ', datetime.datetime.now())

    data_x = data_x[:, masks]
    selected_features_names = np.array(feature_names)[masks]
    print(selected_features_names.shape, selected_features_names)

    df = pd.DataFrame(data_x)
    corr_m = df.corr().abs()
    corr_m = np.array(corr_m).tolist()
    print(corr_m)

    cor_mask = []
    for i in range(opt.selected_features_number):
        cor_mask.append(all([corr_m[i][j] <= opt.feature_correlation_threshold for j in range(i+1, opt.selected_features_number)]))

    cor_mask = np.array(cor_mask)
    data_x = data_x[:, cor_mask]
    final_features_names = selected_features_names[cor_mask]
    print(final_features_names.shape, final_features_names.tolist())

    return data_x, data_y, final_features_names.tolist()

def train_and_predict_svm(data_x, data_y, c, svc_kernel, cross_validation_fold_number):
    average_accuracy_score = 0
    average_auc_score = 0

    train_auc_score = 0
    train_acc_score = 0

    # This cross-validation object is a variation of KFold that returns stratified folds. 
    # The folds are made by preserving the percentage of samples for each class.
    # Stratified sampling (分层抽样)
    for train_index, val_index in StratifiedKFold(n_splits = cross_validation_fold_number).split(data_x, data_y):
        x_train, x_val = data_x[train_index], data_x[val_index]
        y_train, y_val = data_y[train_index], data_y[val_index]
        model = SVC(kernel=svc_kernel, C=c)
        model.fit(x_train, y_train)

        y_decision = model.decision_function(x_val)
        y_predict = model.predict(x_val)

        auc = roc_auc_score(y_val, y_decision)
        average_auc_score += auc / cross_validation_fold_number

        acc = accuracy_score(y_val, y_predict)
        average_accuracy_score += acc / cross_validation_fold_number

        train_auc_score += roc_auc_score(y_train, model.decision_function(x_train)) / cross_validation_fold_number
        train_acc_score += accuracy_score(y_train, model.predict(x_train)) / cross_validation_fold_number

    return average_auc_score, average_accuracy_score, c


def train_svm_all_data(opt, selected_features_names, c, model_path):
    data_x, data_y, feature_names, _ = combine_data(opt.radiomics_parameters_name, is_training = True)
    masks = np.array([name in selected_features_names for name in feature_names])
    data_x = np.array(data_x)[:, masks]
    data_y = np.array(data_y)

    model = SVC(kernel=opt.svc_kernel, C=c)
    model.fit(data_x, data_y)

    y_predict = model.predict(data_x)
    y_decision = model.decision_function(data_x)
    auc = roc_auc_score(data_y, y_decision)
    acc = accuracy_score(data_y, y_predict)

    pickle.dump(model, open(model_path, 'wb'))
    return auc, acc

def test_svm(opt, selected_features_names, model_path, predict_result_path):
    clinical_info_file_path = TESTING_CLINICAL_INFO_FILE_PATH
    _, patient_id_list, _ = get_clinical_data(clinical_info_file_path)
    data_x, data_y, feature_names, _ = combine_data(opt.radiomics_parameters_name, is_training = False)

    masks = np.array([name in selected_features_names for name in feature_names])
    data_x = np.array(data_x)[:, masks]
    data_y = np.array(data_y)

    model = pickle.load(open(model_path, 'rb'))


    y_predict = model.predict(data_x)
    y_decision = model.decision_function(data_x)
    auc = roc_auc_score(data_y, y_decision)
    acc = accuracy_score(data_y, y_predict)

    y_predict = y_predict.tolist()
    index = patient_id_list.index('ACRIN-6698-641246')  # 该patient因为没有同时具有t0和t2的feature所以被去除了，默认是'pCR'
    y_predict = y_predict[0:index] + [1] + y_predict[index:]

    print(len(y_predict), y_predict)
    print(len(patient_id_list), patient_id_list)

    data = {'patient_id': patient_id_list, 'pCR/non-pCR': y_predict}
    pd.DataFrame(data).to_csv(predict_result_path)
    
    return auc, acc

