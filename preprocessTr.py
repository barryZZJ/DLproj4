import nibabel as nib
import SimpleITK as itk
import numpy as np
import time
from nibabel.viewers import OrthoSlicer3D

#固定值截断[-300，300]
def preprocess_state(filename):
	#读取文件
	open_filename = 'data/imagesTr/'+ filename +'.nii.gz'
	save_filename = 'data/imagesTr_Processed/'+ filename +'_Processed'+'.nii.gz'
	raw_img = nib.load(open_filename)
	raw_img_fdata = raw_img.get_fdata()
	data = np.transpose(raw_img_fdata, [2, 1, 0]) #转置
	(z, y, x) = data.shape
	print(x,' ',y,' ',z)

	maxa = data.max()
	mina = data.min()

	maxb = 300
	minb = -300

	# (x-minb)/(maxb-minb) = (value-mina)/(maxa-mina)
	# x = value * (maxb-minb)/(maxa-mina)
	# 比例
	a = (maxb-minb)/(maxa-mina)
	# 截断
	print('start:')
	tot = x * y * z
	count = 0
	for i in range(z):
	    for j in range(x):
	        for k in range(y):
	            value = data[i,j,k]
	            data[i,j,k] = (value-mina) * a + minb
	            count+=1
	            if count % 100000 == 0:
	            	print("当前进度：{}%".format(round(count/tot*100,2)))

    #保存
	img = itk.GetImageFromArray(data)
	itk.WriteImage(img, save_filename)
	print(filename + " 处理完成")


#根据分布动态截断
def preprocess_dynamic(filename):
	#读取文件
	open_filename = 'data/imagesTr/'+ filename +'.nii.gz'
	save_filename = 'data/imagesTr_Processed/'+ filename +'_Processed'+'.nii.gz'
	raw_img = nib.load(open_filename)
	raw_img_fdata = raw_img.get_fdata()
	data = np.transpose(raw_img_fdata, [2, 1, 0]) #转置
	(z, y, x) = data.shape

	#寻找截断集合
	reshape_data = data.ravel()
	sorted_data = sorted(reshape_data)
	#counter = collections.Counter(reshape_data)
	#counter_list = list(counter.elements())
	num_X = round(0.05 * len(sorted_data))
	num_Y = round(0.95 * len(sorted_data))


	maxa = data.max()
	mina = data.min()

	maxb = sorted_data[num_Y]
	minb = sorted_data[num_X]

	print(maxb,' ',minb)

	# (x-minb)/(maxb-minb) = (value-mina)/(maxa-mina)
	# x = value * (maxb-minb)/(maxa-mina)
	# 比例
	a = (maxb-minb)/(maxa-mina)
	# 截断
	print('start:')
	tot = x * y * z
	count = 0
	for i in range(z):
	    for j in range(x):
	        for k in range(y):
	            value = data[i,j,k]
	            data[i,j,k] = (value-mina) * a + minb
	            count+=1
	            if count % 100000 == 0:
	            	print("当前进度：{}%".format(round(count/tot*100,2)))

	print(maxb,' ',minb)

    #保存
	img = itk.GetImageFromArray(data)
	itk.WriteImage(img, save_filename)
	print(filename + " 处理完成")



if __name__ == '__main__':
	time_start=time.time()
	for i in range(131):
		preprocess_state('liver_{}'.format(i))
		print('总进度：{}%'.format(round(i/130*100,2)))
	time_end=time.time()
	print('totally cost (min):',(time_end-time_start)/60)