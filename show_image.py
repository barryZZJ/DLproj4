import nibabel as nib
import SimpleITK as itk
import numpy as np
from nibabel.viewers import OrthoSlicer3D

#展示3d图像
def show_image(filepath):
	img = nib.load(filepath)
	OrthoSlicer3D(img.dataobj).show()

if __name__ == '__main__':
	filepath = 'data/imagesTr_Processed/liver_3_Processed.nii.gz'
	show_image(filepath)
