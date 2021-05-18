import nibabel as nib
import SimpleITK as itk
import numpy as np
import time
from nibabel.viewers import OrthoSlicer3D

def preprocess_label(filename):
	#读取文件
	open_filename = 'data/labelsTr/'+ filename +'.nii.gz'
	save_filename = 'data/labelsTr_Processed/'+ filename +'_Labels_Processed'+'.nii.gz'
	raw_img = nib.load(open_filename)
	raw_img_fdata = raw_img.get_fdata()
	data = np.transpose(raw_img_fdata, [2, 1, 0]) #转置
	(z, y, x) = data.shape
	print(x,' ',y,' ',z)

	# 截断
	print('start:')
	tot = x * y * z
	count = 0
	for i in range(z):
	    for j in range(x):
	        for k in range(y):
	            value = data[i,j,k]
	            if value == 2:
	            	data[i,j,k] = 1
	            count+=1
	            if count % 100000 == 0:
	            	print("当前进度：{}%".format(round(count/tot*100,2)))

    #保存
	img = itk.GetImageFromArray(data)
	itk.WriteImage(img, save_filename)
	print(filename + " 处理完成")

if __name__ == '__main__':
	time_start=time.time()
	for i in range(131):
		preprocess_label('liver_{}'.format(i))
		print('总进度：{}%'.format(round(i/130*100,2)))
	time_end=time.time()
	print('totally cost (min):',(time_end-time_start)/60)