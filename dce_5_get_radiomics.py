import dicom2nifti
import os
import radiomics
import yaml
import pydicom
import pydicom_seg
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import six
import pickle
import shutil
import natsort
import matplotlib

reverse_mask_t0_name_list = ['ACRIN-6698-342959', 'ACRIN-6698-587604', 'ACRIN-6698-687288', 'ACRIN-6698-689480', 'ACRIN-6698-728182', 'ACRIN-6698-839274'] # cases whose analysis mask needs to flip
reverse_mask_t2_name_list = ['ACRIN-6698-409288', 'ACRIN-6698-587604', 'ACRIN-6698-687288', 'ACRIN-6698-728182''ACRIN-6698-839274'] # cases whose analysis mask needs to flip
null_analysis_mask_t2_name_list = ['ACRIN-6698-298101', 'ACRIN-6698-417474', 'ACRIN-6698-597907', 'ACRIN-6698-913527', 'ACRIN-6698-929980', 'ACRIN-6698-641246'] # cases whose analysis mask in T2 is all zero, so can't get radiomics

radiomics_parameters_name = 'ParamSetting5_2dFalse_normFalse_binCount100'
para_radiomics_yaml_path = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/{}.yaml'.format(radiomics_parameters_name)

training_folder = '/Users/ranyan/workspace/BMMR2/data/manifest-Training-Set/ACRIN-6698'
testing_folder = '/Users/ranyan/workspace/BMMR2/data/manifest-Testing-Set/ACRIN-6698'
mydata_training_folder_curr = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/{}/training_data'.format(radiomics_parameters_name)
mydata_testing_folder_curr = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/{}/testing_data'.format(radiomics_parameters_name)
mydata_training_folder = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/training_data'
mydata_testing_folder = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/testing_data'
temp_folder = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/temp2'
temp_name_list = ['ACRIN-6698-107700']


is_training = False
image_name_keyword = 'DCE-'   # Choose from 'DCE-', 'SER-', 'PE2-', 'PE56-'
image_phase_select = 2  #Choose from (0, 2)
scan_time_keyword = 'T2-'  # Choose from T0-, T1-, T2-
start_patient = 0

is_plot = False
mask_name_keyword = 'Analysis Mask'  #Choose from Analysis Mask

image_name_suffix = image_name_keyword[:-1].lower()
if image_name_keyword == 'DCE-':
    image_name_suffix = image_name_suffix + str(image_phase_select)  # Choose from dce0, dce2, ser, pe2, pe56
scan_time_suffix = scan_time_keyword[:-1].lower() # Choose from t0, t1, t2

training_or_tesing_folder = training_folder if is_training else testing_folder
mydata_training_or_tesing_folder = mydata_training_folder if is_training else mydata_testing_folder
mydata_training_or_tesing_folder_curr = mydata_training_folder_curr if is_training else mydata_testing_folder_curr

patient_name_list = [x for x in os.listdir(training_or_tesing_folder) if 'DS_Store' not in x]
patient_name_list.sort()

for i in range(start_patient,len(patient_name_list)):
    patient_name = patient_name_list[i]
    print("------ patient: {}------".format(patient_name))

    patient_name_folder = os.path.join(training_or_tesing_folder,patient_name)
    img_nifti_path = os.path.join(mydata_training_or_tesing_folder,'img_{}_{}'.format(image_name_suffix, scan_time_suffix),"{}.nii".format(patient_name))
    mask_nifti_path = os.path.join(mydata_training_or_tesing_folder,'analysisMask_tumor_{}'.format(scan_time_suffix),"{}.nii".format(patient_name))
    features_path = os.path.join(mydata_training_or_tesing_folder_curr,'features_{}_tumor_{}'.format(image_name_suffix, scan_time_suffix),"{}.p".format(patient_name))
    features_suffix = "_{}_tumor_{}".format(image_name_suffix, scan_time_suffix)

    if patient_name in null_analysis_mask_t2_name_list and scan_time_keyword == 'T2-' and mask_name_keyword == 'Analysis Mask':
        print("skip patient since null analysis mask in T2")
        continue


    for time_folder in os.listdir(patient_name_folder):
        if scan_time_keyword in time_folder:  # Choose time (T0,T1, orT2) image folder
            break
    time_folder = os.path.join(patient_name_folder,time_folder)

    for allimage_folder in os.listdir(time_folder):
        if image_name_keyword == 'PE56-':
            if 'PE5-' in allimage_folder or 'PE6-' in allimage_folder or  'PE4-' in allimage_folder:
                img_folder = os.path.join(time_folder,allimage_folder)
        else:
            if image_name_keyword in allimage_folder:
                img_folder = os.path.join(time_folder, allimage_folder)
        if mask_name_keyword in allimage_folder:
            mask_folder = os.path.join(time_folder,allimage_folder)

    img_file_list = [x for x in os.listdir(img_folder) if 'DS_Store' not in x]
    img_file_list_natsorted = natsort.natsorted(img_file_list)
    img_first_file = os.path.join(img_folder,img_file_list_natsorted[0]) #select the 1st dicom file
    img_second_file = os.path.join(img_folder,img_file_list_natsorted[1]) #select the 2nd dicom file
    ds = pydicom.dcmread(img_first_file)
    ds2 = pydicom.dcmread(img_second_file)
    patient_orientation = ds.ImageOrientationPatient
    slice_num = int(ds[0x0117, 0x1093][2])  # number of slices
    img_phase_num = int(len(img_file_list_natsorted) / slice_num)

    if 'bi-lateral' in ds.SeriesDescription or hasattr(ds, 'TemporalResolution') or hasattr(ds2, 'TemporalResolution'):
        slices = img_file_list_natsorted[image_phase_select:len(img_file_list_natsorted):img_phase_num] # Save the slices with the number of image phases as the interval
    else:
        slices = img_file_list_natsorted[image_phase_select * slice_num : (image_phase_select + 1) * slice_num]  # save No.1 to No.slice_num slices
    slice_num_plot1 = int(slice_num * 0.5)
    slice_num_plot2 = int(slice_num * 0.7)

    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)
    for slice in slices:
        shutil.copyfile(os.path.join(img_folder,slice),os.path.join(temp_folder,slice))

    # image: dicom to nifti
    dicom2nifti.convert_dicom.dicom_series_to_nifti(temp_folder, output_file = img_nifti_path)
    img_nifti = nib.load(img_nifti_path)
    img_array = np.array(img_nifti.dataobj)
    os.remove(img_nifti_path)
    if is_plot:
        plt.figure(1)
        plt.subplot(221)
        plt.imshow(img_array[:, :, slice_num_plot1])
        plt.subplot(222)
        plt.imshow(img_array[:, :, slice_num_plot2])
    img_nifti = nib.Nifti1Image(img_array, None)
    nib.save(img_nifti,img_nifti_path)

    '''
    # mask: dicom to nifti
    mask_array = pydicom.dcmread(os.path.join(mask_folder, "1-1.dcm"))
    reader = pydicom_seg.SegmentReader()
    result = reader.read(mask_array)
    mask_array = result.segment_data(1)  # Assume segment_number in result.available_segments is only 1
    mask_array = np.transpose(mask_array, (1, 2, 0))

    if mask_array.shape[2] - img_array.shape[2] == 1:
        mask_array = mask_array[:,:,:img_array.shape[2]] # if analysis mask has one more slice than image, remove the last slice

    standard_patient_orientation_list = [[-1, -0, 0, -0, -1, 0],  # needs to rot mask for 90 degree 1 time
                                [ 1,  0, 0,  0,  1, 0]] # needs to rot mask for 90 degree 3 times

    standard_patient_orientation = standard_patient_orientation_list[0]
    if all([abs(patient_orientation[i] - standard_patient_orientation[i]) < 0.01 for i in range(len(standard_patient_orientation))]):
        mask_array = np.rot90(mask_array, k = 1)

    standard_patient_orientation = standard_patient_orientation_list[1]
    if all([abs(patient_orientation[i] - standard_patient_orientation[i]) < 0.01 for i in range(len(standard_patient_orientation))]):
        mask_array = np.rot90(mask_array, k = 3)

    if patient_name in reverse_mask_t0_name_list and scan_time_keyword == 'T0-':
        mask_array = np.flip(mask_array, axis = 2) # flip along z axis for special cases
    if patient_name in reverse_mask_t2_name_list and scan_time_keyword == 'T2-':
        mask_array = np.flip(mask_array, axis = 2) # flip along z axis for special cases

    if is_plot:
        plt.subplot(223)
        plt.imshow(mask_array[:, :, slice_num_plot1])
        plt.subplot(224)
        plt.imshow(mask_array[:, :, slice_num_plot2])
        plt.show()

    mask_array[mask_array == 0] = -1  # Use only mask value 0 as the tumor segmentation that was used in the primary analysis. This is the "SER" Functional Tumor Volume.
    mask_array[mask_array > 0] = 0  # Set other mask value to 0
    mask_array[mask_array < 0] = 1  # Set tumor segmentation mask value to 1
    plt.figure(2)
    plt.imshow(mask_array[:, :, slice_num_plot1])
    plt.show()
    mask_nifti = nib.Nifti1Image(mask_array, None)
    nib.save(mask_nifti, mask_nifti_path)
    '''

    # mask: load nifti
    mask_nifti = nib.load(mask_nifti_path)
    mask_array = np.array(mask_nifti.dataobj)
    if is_plot:
        plt.subplot(223)
        plt.imshow(mask_array[:, :, slice_num_plot1])
        plt.subplot(224)
        plt.imshow(mask_array[:, :, slice_num_plot2])
        plt.show()


    # Radiomics
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(para_radiomics_yaml_path)
    radiomics_features = extractor.execute(img_nifti_path, mask_nifti_path)

    # save radiomics features
    radiomics_features_dic = {}
    for key, value in six.iteritems(radiomics_features):
        radiomics_features_dic[key+features_suffix] = value

    pickle.dump(radiomics_features_dic,open(features_path, 'wb'))

print('--------FINISHED: isTraining_{}{}-------'.format(is_training, features_suffix))

