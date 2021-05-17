import nibabel as nib
import SimpleITK as itk
import numpy as np
import time
from nibabel.viewers import OrthoSlicer3D

a = []
# 统计图像尺寸
def deal(filename):
	open_filename = 'data/imagesTr/'+ filename +'.nii.gz'
	save_filename = 'data/imagesTr_Processed/'+ filename +'_Processed'+'.nii.gz'
	raw_img = nib.load(open_filename)
	raw_img_fdata = raw_img.get_fdata()
	data = np.transpose(raw_img_fdata, [2, 1, 0]) #转置
	(z, y, x) = data.shape
	a.append([x,y,z])

if __name__ == '__main__':
	for i in range(131):
		deal('liver_{}'.format(i))
		print('总进度：{}%'.format(round(i/130*100,2)))
	with open('statistics_result.txt', 'w') as f:
		count = 0
		for [x,y,z] in a:
			f.write('liver_{}:{} {} {}\n'.format(count,x,y,z))
			print('liver_{}:{} {} {}'.format(count,x,y,z))
			count+=1
	print('统计完成')
