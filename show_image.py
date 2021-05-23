import nibabel as nib
import SimpleITK as itk
import numpy as np
from nibabel.viewers import OrthoSlicer3D

#展示3d图像
def show_image(filepath):
	img = nib.load(filepath)
	OrthoSlicer3D(img.dataobj).show()

#展示最终分割结果
def show_result_image(image_data,label_data):
	(x, y, z) = label_data.shape
	for i in range(x):
		for j in range(y):
			for k in range(z):
				if label_data[i,j,k] == 1:
					image_data[i,j,k] = 300
	# 保存
	img = itk.GetImageFromArray(image_data)
	itk.WriteImage(img, 'data/temp/show_temp.nii.gz')

	#展示
	show_image('data/temp/show_temp.nii.gz')

if __name__ == '__main__':
	# filepath = 'data/imagesTr_Processed/liver_3_Processed.nii.gz'
	# show_image(filepath)
	open_images_path =  'data/imagesTr_lr/liver_3_lr.nii.gz'
	open_labels_path = 'data/labelsTr_lr/liver_3_labels_lr.nii.gz'
	raw_img = nib.load(open_images_path)
	raw_lab = nib.load(open_labels_path)
	raw_img_fdata = raw_img.get_fdata()
	raw_lab_fdata = raw_lab.get_fdata()
	image_data = np.transpose(raw_img_fdata, [2, 1, 0])
	label_data = np.transpose(raw_lab_fdata, [2, 1, 0])
	show_result_image(image_data, label_data)

