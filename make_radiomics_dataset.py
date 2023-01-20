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

reverse_analysis_mask_name_list = ['ACRIN-6698-342959', 'ACRIN-6698-839274', 'ACRIN-6698-587604', 'ACRIN-6698-687288', 'ACRIN-6698-689480', 'ACRIN-6698-728182'] # cases whose analysis mask needs to flip
para_radiomics_yaml_path = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/ParamSetting4_2dTrue_normFalse_binWidth15.yaml'

training_folder = '/Users/ranyan/workspace/BMMR2/data/manifest-Training-Set/ACRIN-6698'
testing_folder = '/Users/ranyan/workspace/BMMR2/data/manifest-Testing-Set/ACRIN-6698'
mydata_folder = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/mydata/training_data'
temp_folder = '/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/temp'
temp_name_list = ['ACRIN-6698-467686']

plot_flag = True

patient_name_list = [x for x in os.listdir(training_folder) if 'DS_Store' not in x]
patient_name_list.sort()

for i in range(0,len(temp_name_list)):
    patient_name = temp_name_list[i]
    print("------ patient: {}------".format(patient_name))

    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)
    os.mkdir(temp_folder)

    patient_name_folder = os.path.join(training_folder,patient_name)
    dce_nifti_path = os.path.join(mydata_folder,'img_dce0_t2',"{}.nii".format(patient_name))
    analysis_mask_nifti_path = os.path.join(mydata_folder,'analysisMask_tumor_t2',"{}.nii".format(patient_name))
    dce_features_path = os.path.join(mydata_folder,'features_dce0_tumor_t2',"{}.p".format(patient_name))


    for time_folder in os.listdir(patient_name_folder):
        if 'T2-' in time_folder:  # Choose time (T0,T1, orT2) image folder
            break
    time_folder = os.path.join(patient_name_folder,time_folder)

    for image_folder in os.listdir(time_folder):
        if 'original DCE' in image_folder:  # Choose original DCE
            dce_folder = os.path.join(time_folder,image_folder)
        if 'Analysis Mask' in image_folder: # Choose Analysis Mask
            analysis_mask_folder = os.path.join(time_folder,image_folder)

    dce_file_list = [x for x in os.listdir(dce_folder) if 'DS_Store' not in x]
    dce_file_list_natsorted = natsort.natsorted(dce_file_list)
    dce_first_file = os.path.join(dce_folder,dce_file_list_natsorted[0]) #select the 1st dce dicom file
    ds = pydicom.dcmread(dce_first_file)
    patient_orientation = ds.ImageOrientationPatient
    slice_num = ds[0x0117, 0x1093][2]  # number of slices in the first dce phase
    dce_phase_num = int(len(dce_file_list_natsorted) / slice_num)

    if 'bi-lateral' in ds.SeriesDescription or hasattr(ds, 'TemporalResolution'):
        slices = dce_file_list_natsorted[0:len(dce_file_list_natsorted):dce_phase_num] # Save the slices with the number of dce phases as the interval
    else:
        slices = dce_file_list_natsorted[:slice_num]  # save No.1 to No.slice_num slices
    slice_num_plot1 = int(slice_num * 0.3)
    slice_num_plot2 = int(slice_num * 0.7)

    for slice in slices:
        shutil.copyfile(os.path.join(dce_folder,slice),os.path.join(temp_folder,slice))

    # image: dicom to nifti
    dicom2nifti.convert_dicom.dicom_series_to_nifti(temp_folder, output_file = dce_nifti_path)
    dce_nifti = nib.load(dce_nifti_path)
    dce_array = np.array(dce_nifti.dataobj)
    os.remove(dce_nifti_path)
    if plot_flag:
        plt.figure(1)
        plt.subplot(221)
        plt.imshow(dce_array[:, :, slice_num_plot1])
        plt.subplot(222)
        plt.imshow(dce_array[:, :, slice_num_plot2])
    dce_nifti = nib.Nifti1Image(dce_array, None)
    nib.save(dce_nifti,dce_nifti_path)

    # mask: dicom to nifti
    analysis_mask = pydicom.dcmread(os.path.join(analysis_mask_folder, "1-1.dcm"))
    reader = pydicom_seg.SegmentReader()
    result = reader.read(analysis_mask)
    analysis_mask = result.segment_data(1)  # Assume segment_number in result.available_segments is only 1
    analysis_mask = np.transpose(analysis_mask, (1, 2, 0))

    if analysis_mask.shape[2] - dce_array.shape[2] == 1:
        analysis_mask = analysis_mask[:,:,:dce_array.shape[2]] # if analysis mask has one more slice than dce, remove the last slice

    standard_patient_orientation_list = [[-1, -0, 0, -0, -1, 0],  # needs to rot mask for 90 degree 1 time
                                [ 1,  0, 0,  0,  1, 0]] # needs to rot mask for 90 degree 3 times

    standard_patient_orientation = standard_patient_orientation_list[0]
    if all([abs(patient_orientation[i] - standard_patient_orientation[i]) < 0.01 for i in range(len(standard_patient_orientation))]):
        analysis_mask = np.rot90(analysis_mask, k = 1)

    standard_patient_orientation = standard_patient_orientation_list[1]
    if all([abs(patient_orientation[i] - standard_patient_orientation[i]) < 0.01 for i in range(len(standard_patient_orientation))]):
        analysis_mask = np.rot90(analysis_mask, k = 3)

    if patient_name in reverse_analysis_mask_name_list:
        analysis_mask = np.flip(analysis_mask, axis = 2) # flip along z axis for special cases

    if plot_flag:
        plt.subplot(223)
        plt.imshow(analysis_mask[:, :, slice_num_plot1])
        plt.subplot(224)
        plt.imshow(analysis_mask[:, :, slice_num_plot2])
        plt.show()
    analysis_mask[analysis_mask == 0] = -1  # Use only mask value 0 as the tumor segmentation that was used in the primary analysis. This is the "SER" Functional Tumor Volume.
    analysis_mask[analysis_mask > 0] = 0  # Set other mask value to 0
    analysis_mask[analysis_mask < 0] = 1  # Set tumor segmentation mask value to 1
    # plt.figure(2)
    # plt.imshow(analysis_mask[:, :, slice_num_plot1])
    # plt.show()
    analysis_mask_nifti = nib.Nifti1Image(analysis_mask, None)
    nib.save(analysis_mask_nifti, analysis_mask_nifti_path)


    # Radiomics
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(para_radiomics_yaml_path)
    features_dce = extractor.execute(dce_nifti_path, analysis_mask_nifti_path)

    # save radiomics features
    dce_data = {}
    for key, value in six.iteritems(features_dce):
        dce_data[key+"_dce0_tumor_t2"] = value

 #   pickle.dump(dce_data,open(dce_features_path, 'wb'))


