from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
import numpy as np
import pickle
import pandas as pd
import os
import datetime
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.linear_model import LassoCV, Lasso
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import scale
# from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.utils._testing import ignore_warnings
import matplotlib.pyplot as plt


SELECTED_FEATURES_NUMBER = 10
FEATURE_NAME_KEY_WORDS = ['firstorder', 'original_shape', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm']
FEATURES_FOLDER_PATH = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/ParamSetting3_2dFalse_normFalse_binWidth15/training_data/'
TRAINING_CLINICAL_INFO_FILE_PATH = '/Users/ranyan/workspace/BMMR2/data/BMMR2_Training_set_cases_with_clinical_data_20210520.csv'
TESTING_CLINICAL_INFO_FILE_PATH = '/Users/ranyan/workspace/BMMR2/data/BMMR2_Testing_set_cases_with_clinical_data_20210520.csv'
SVM_MODEL_SAVED_PATH = ''
CROSS_VALIDATION_FOLD_NUMBER = 5
SFFS_FEATURES_30 = ['SBRgrade_I (Low)', 'age', 'hrher4g_HR + / HER2 -', 'hrher4g_HR - / HER2 +', 'original_firstorder_10Percentile_dce', 'original_firstorder_Mean_dce', 'original_firstorder_Minimum_dce', 'original_firstorder_RootMeanSquared_dce', 'original_glcm_DifferenceAverage_dce', 'original_glcm_Id_dce', 'original_glcm_Idm_dce', 'original_glcm_Idn_dce', 'original_gldm_DependenceNonUniformityNormalized_dce', 'original_gldm_DependenceVariance_dce', 'original_gldm_LargeDependenceEmphasis_dce', 'original_gldm_SmallDependenceEmphasis_dce', 'original_glrlm_LongRunEmphasis_dce', 'original_glrlm_RunPercentage_dce', 'original_glrlm_RunVariance_dce', 'original_glszm_LargeAreaLowGrayLevelEmphasis_dce', 'original_glszm_SizeZoneNonUniformityNormalized_dce', 'original_glszm_SmallAreaEmphasis_dce', 'original_glszm_ZonePercentage_dce', 'original_shape_LeastAxisLength_dce', 'original_shape_MajorAxisLength_dce', 'original_shape_Maximum2DDiameterColumn_dce', 'original_shape_Maximum2DDiameterRow_dce', 'original_shape_Maximum2DDiameterSlice_dce', 'original_shape_Maximum3DDiameter_dce', 'original_shape_MinorAxisLength_dce']
SFFS_FEATURES_20 = ['SBRgrade_I (Low)', 'age', 'hrher4g_HR + / HER2 -', 'hrher4g_HR - / HER2 +', 'original_firstorder_10Percentile_dce', 'original_firstorder_Minimum_dce', 'original_firstorder_RootMeanSquared_dce', 'original_glcm_Id_dce', 'original_glcm_Idm_dce', 'original_gldm_DependenceNonUniformityNormalized_dce', 'original_gldm_DependenceVariance_dce', 'original_gldm_LargeDependenceEmphasis_dce', 'original_gldm_SmallDependenceEmphasis_dce', 'original_glrlm_RunPercentage_dce', 'original_glszm_SizeZoneNonUniformityNormalized_dce', 'original_glszm_SmallAreaEmphasis_dce', 'original_glszm_ZonePercentage_dce', 'original_shape_LeastAxisLength_dce', 'original_shape_MajorAxisLength_dce', 'original_shape_Maximum2DDiameterRow_dce']
SFFS_FEATURES_10 = ['SBRgrade_I (Low)', 'age', 'hrher4g_HR + / HER2 -', 'hrher4g_HR - / HER2 +', 'original_firstorder_Minimum_dce', 'original_glcm_Id_dce', 'original_glcm_Idm_dce', 'original_gldm_DependenceNonUniformityNormalized_dce', 'original_shape_LeastAxisLength_dce', 'original_shape_Maximum2DDiameterRow_dce']

# t0的min可能为零
DELETE_FEATURES = ['firstorder_Minimum']

FEATURE_PREFIX_CATEGORY_SET = list(set([category[:-2] for category in os.listdir(FEATURES_FOLDER_PATH) if 'DS_Store' not in category]))
T_SUFFIX = ['t0', 't2']

SER_FEATURE_ZERO_VALUES = {'original_ngtdm_Complexity_ser_tumor_', 'original_glcm_ClusterShade_ser_tumor_', 'original_glcm_Imc1_ser_tumor_', 'original_ngtdm_Contrast_ser_tumor_', 'original_glcm_DifferenceVariance_ser_tumor_', 'original_glcm_SumSquares_ser_tumor_', 'original_glcm_ClusterTendency_ser_tumor_', 'original_glrlm_GrayLevelVariance_ser_tumor_', 'original_glcm_Contrast_ser_tumor_', 'original_glcm_ClusterProminence_ser_tumor_', 'original_glcm_InverseVariance_ser_tumor_', 'original_glcm_Imc2_ser_tumor_', 'original_ngtdm_Strength_ser_tumor_', 'original_glszm_GrayLevelVariance_ser_tumor_', 'original_gldm_GrayLevelVariance_ser_tumor_', 'original_ngtdm_Busyness_ser_tumor_', 'original_glcm_DifferenceAverage_ser_tumor_'}


def get_clinical_data(clinical_data_path):
    csv_data = pd.read_csv(clinical_data_path)
    names = csv_data.columns
    for col in range(5, 9):
        prefix = names[col]
        dummy_feature = pd.get_dummies(csv_data[prefix], prefix=prefix)
        csv_data = pd.concat([csv_data, dummy_feature], axis=1)
        csv_data.drop([prefix], axis=1, inplace=True)

    patient_id_list = csv_data[names[1]].to_list()
    labels = [1 if label == 'pCR' else 0 for label in csv_data[names[10]].to_list()]

    csv_data.drop([names[0], names[1], names[2], names[3], names[10], names[11], 'race_Unknown'],
                  axis=1, inplace=True)

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


def is_both_t0_t2_exist(patient_id):
    for category in FEATURE_PREFIX_CATEGORY_SET:
        for t in T_SUFFIX:
            path = FEATURES_FOLDER_PATH + category + t + '/' + patient_id + '.p'
            if not os.path.exists(path):
                return False
    return True


def get_all_feature_data_for_single_patient(patient_id):
    feature_names = []
    feature_values = []

    for feature_prefix in FEATURE_PREFIX_CATEGORY_SET:
        t0_feature_file = FEATURES_FOLDER_PATH + feature_prefix + 't0/' + patient_id + '.p'
        t2_feature_file = FEATURES_FOLDER_PATH + feature_prefix + 't2/' + patient_id + '.p'
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


def combine_data():

    clinical_data, patient_id_list, labels = get_clinical_data(TRAINING_CLINICAL_INFO_FILE_PATH)
    clinical_feature_names = clinical_data.columns.tolist()
    patient_id_list = [patient_id for patient_id in patient_id_list if is_both_t0_t2_exist(patient_id)]
    # 因为算t0和算t2的病人有不一样的
    number_of_data = len(patient_id_list)
    number_of_clinical_features = len(clinical_feature_names)
    print(number_of_clinical_features)
    clinical_data = clinical_data.to_numpy()

    data_x = []
    data_y = []
    all_feature_names = []


    for i in range(number_of_data):
        patient_id = patient_id_list[i]
        tmp_feature_names, feature_values = get_all_feature_data_for_single_patient(patient_id)

        for j in range(number_of_clinical_features):
            tmp_feature_names.append(clinical_feature_names[j])
            feature_values.append(clinical_data[i][j])

        all_feature_names, feature_values = zip(*sorted(zip(tmp_feature_names, feature_values)))
        data_x.append(feature_values)
        data_y.append(labels[i])

    data_x = scale(np.array(data_x))
    data_y = np.array(data_y)

    return data_x, data_y, all_feature_names, patient_id_list


@ignore_warnings(category=ConvergenceWarning)
def select_features():
    data_x, data_y, feature_names, _ = combine_data()

    lasso = Lasso(alpha=0.003).fit(data_x, data_y)
    importance = np.abs(lasso.coef_)
    sorted_importance = sorted(importance)
    threshold = sorted_importance[-100]
    print(sorted_importance)

    first_mask = np.array([v > threshold for v in importance])
    data_x = data_x[:, first_mask]
    feature_names = np.array(feature_names)[first_mask]
    print(feature_names)
    print(data_x.shape)

    # fig, (ax1, ax2) = plt.subplots(2,1)
    # ax1.plot(range(334), importance)
    # ax2.imshow(corr_m, aspect = 'auto')
    # plt.show()

    print('feature select starts at: ', datetime.datetime.now())
    lasso_cv = LassoCV(tol=0.002, n_jobs=-1).fit(data_x, data_y)
    sfs_forward = SequentialFeatureSelector(
        lasso_cv, n_features_to_select=SELECTED_FEATURES_NUMBER, direction="forward"
    ).fit(data_x, data_y)

    masks = sfs_forward.get_support()

    print('feature select ends at: ', datetime.datetime.now())

    # masks = np.array([name in DELTA_FEATURE_30 for name in feature_names])
    data_x = data_x[:, masks]

    df = pd.DataFrame(data_x)
    corr_m = df.corr().abs()
    corr_m = np.array(corr_m).tolist()
    print(corr_m)

    cor_mask = []
    for i in range(SELECTED_FEATURES_NUMBER):
        cor_mask.append(all([corr_m[i][j] <= 0.95 for j in range(i+1, SELECTED_FEATURES_NUMBER)]))

    cor_mask = np.array(cor_mask)
    data_x = data_x[:, cor_mask]
    selected_feature_names = np.array(feature_names)[masks]
    print(selected_feature_names.shape, selected_feature_names)
    final_features = selected_feature_names[cor_mask]
    print(final_features.shape, final_features.tolist())

    return data_x, data_y


def train_and_predict_svm(data_x, data_y, c):
    average_accuracy_score = 0
    average_auc_score = 0

    train_auc_score = 0
    train_acc_score = 0

    # This cross-validation object is a variation of KFold that returns stratified folds. 
    # The folds are made by preserving the percentage of samples for each class.
    # Stratified sampling (分层抽样)
    for train_index, val_index in StratifiedKFold(n_splits=CROSS_VALIDATION_FOLD_NUMBER).split(data_x, data_y):
        x_train, x_val = data_x[train_index], data_x[val_index]
        y_train, y_val = data_y[train_index], data_y[val_index]
        model = SVC(kernel='linear', C=c)
        model.fit(x_train, y_train)

        y_decision = model.decision_function(x_val)
        y_predict = model.predict(x_val)

        auc = roc_auc_score(y_val, y_decision)
        average_auc_score += auc / CROSS_VALIDATION_FOLD_NUMBER

        acc = accuracy_score(y_val, y_predict)
        average_accuracy_score += acc / CROSS_VALIDATION_FOLD_NUMBER

        train_auc_score += roc_auc_score(y_train, model.decision_function(x_train)) / CROSS_VALIDATION_FOLD_NUMBER
        train_acc_score += accuracy_score(y_train, model.predict(x_train)) / CROSS_VALIDATION_FOLD_NUMBER

    return average_auc_score, average_accuracy_score, c


def train_all_data(c):
    data_x, data_y, feature_names, _ = combine_data()
    selected_features = ['Ltype_Multiple masses', 'MRLD', 'hrher4g_HR - / HER2 +', 'original_firstorder_Entropy_ser_tumor_t0', 'original_glcm_ClusterShade_dce2_tumor_t0', 'original_gldm_LargeDependenceHighGrayLevelEmphasis_dce2_tumor_delta', 'original_glszm_HighGrayLevelZoneEmphasis_dce2_tumor_t2', 'original_shape_Flatness_dce0_tumor_t0', 'original_shape_LeastAxisLength_dce0_tumor_delta', 'race_Black']

    masks = np.array([name in selected_features for name in feature_names])
    data_x = np.array(data_x)[:, masks]
    data_y = np.array(data_y)

    model = SVC(kernel='linear', C=c)
    model.fit(data_x, data_y)

    y_predict = model.predict(data_x)
    y_decision = model.decision_function(data_x)
    auc = roc_auc_score(data_y, y_decision)
    acc = accuracy_score(data_y, y_predict)

    print(auc, acc)

    # example: 'model_ParamSetting1_features10_c1.p'
    model_path = '/Users/jacob/Downloads/model_ParamSetting7_features10_c0.3.p'
    pickle.dump(model, open(model_path, 'wb'))


def test():
    _, patient_list, _ = get_clinical_data(TESTING_CLINICAL_INFO_FILE_PATH)
    data_x, _, feature_names, _ = combine_data()
    selected_features = ['Ltype_Multiple masses', 'MRLD', 'hrher4g_HR - / HER2 +', 'original_firstorder_Entropy_ser_tumor_t0', 'original_glcm_ClusterShade_dce2_tumor_t0', 'original_gldm_LargeDependenceHighGrayLevelEmphasis_dce2_tumor_delta', 'original_glszm_HighGrayLevelZoneEmphasis_dce2_tumor_t2', 'original_shape_Flatness_dce0_tumor_t0', 'original_shape_LeastAxisLength_dce0_tumor_delta', 'race_Black']

    masks = np.array([name in selected_features for name in feature_names])
    data_x = np.array(data_x)[:, masks]

    model_path = '/Users/jacob/Downloads/model_ParamSetting7_features10_c0.3.p'
    model = pickle.load(open(model_path, 'rb'))

    y_predict = model.predict(data_x).tolist()
    index = patient_list.index('ACRIN-6698-641246')
    y_predict = y_predict[0:index] + [1] + y_predict[index:]

    print(len(y_predict), y_predict)
    print(len(patient_list), patient_list)

    data = {'patient_id': patient_list, 'pCR/non-pCR': y_predict}
    pd.DataFrame(data).to_csv('/Users/jacob/Downloads/predictResults_ParamSetting7_features10_c0.3.csv')


def main():

    print(FEATURES_FOLDER_PATH)
    # select_features()
    c_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    pool = ThreadPool(len(c_list))
    threads = []
    data_x, data_y = select_features()
    for c in c_list:
        threads.append(pool.apply_async(train_and_predict_svm, args=(data_x, data_y, c)))

    pool.close()
    pool.join()
    results = [p.get() for p in threads]
    for auc, acc, c in results:
        print(c, auc, acc)
    max_auc, max_acc, best_c = sorted(results, reverse=True)[0]
    print('Best c {}, max auc {}, max acc {}'.format(best_c, max_auc, max_acc))

data_x, data_y, feature_names, patient_id_list = combine_data()
data = {'patient_id': patient_id_list, 'pCR/non-pCR': data_y}
for i in range(len(feature_names)):
    data[feature_names[i]] = data_x[:,i]
pd.DataFrame(data).to_csv('/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/features.csv')

a=3