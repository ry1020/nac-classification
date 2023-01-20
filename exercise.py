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

slice_num = 30

# image: dicom to nifti
dcm_folder_path = "/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/dcm"
nifti_folder_path = "/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti"
dicom2nifti.convert_directory(dcm_folder_path, nifti_folder_path)
img = nib.load('/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti/61800_ispy2_volser_uni-lateral_cropped_original_dce.nii.gz')
a = np.array(img.dataobj)
os.remove('/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti/61800_ispy2_volser_uni-lateral_cropped_original_dce.nii.gz')
img_nifti = nib.Nifti1Image(a,None)
nib.save(img_nifti,'/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti/61800_ispy2_volser_uni-lateral_cropped_original_dce.nii')


# load image in nifti
img = nib.load('/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti/61800_ispy2_volser_uni-lateral_cropped_original_dce.nii.gz')
img.shape
a = np.array(img.dataobj)
a.shape
imgplot = plt.imshow(a[:, :, slice_num])

# mask: dicom to nifti
dcm = pydicom.dcmread("/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/mask/1-1.dcm")
reader = pydicom_seg.SegmentReader()
result = reader.read(dcm)
mask_data = result.segment_data(1)  #Assume segment_number in result.available_segments is only 1
mask_data.shape
mask_data = np.transpose(mask_data,(1,2,0))
mask_data = np.rot90(mask_data)
imgplot = plt.imshow(mask_data[:, :, slice_num])
mask_data.shape
mask_data[mask_data == 0] = -1  # Use only mask value 0 as the tumor segmentation that was used in the primary analysis. This is the "SER" Functional Tumor Volume.
mask_data[mask_data > 0] = 0 # Set other mask value to 0
mask_data[mask_data < 0] = 1 # Set tumor segmentation mask value to 1
imgplot = plt.imshow(mask_data[:, :, slice_num])
mask_data_nifti = nib.Nifti1Image(mask_data,None)
nib.save(mask_data_nifti,'/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti_mask/1-1.nii')

# load mask in nifti
img = nib.load('/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti_mask/1-1.nii')
img.shape
a = np.array(img.dataobj)
a.shape
imgplot = plt.imshow(a[:, :, slice_num])


# Radiomics
extractor = radiomics.featureextractor.RadiomicsFeatureExtractor("/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/ParamSetting_0.yaml")
result_t2 = extractor.execute("/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti/61800_ispy2_volser_uni-lateral_cropped_original_dce.nii","/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/nifti_mask/1-1.nii")

# save radiomics features
t2_data = {}
for key,value in six.iteritems(result_t2):
    t2_data[key] = value
pickle.dump(t2_data, open("/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/features/example.p", 'wb'))

#
dic = pickle.load(open("/Users/ranyan/workspace/BMMR2/Project_Ran_radiomics/example_data/features/example.p", 'rb'))
