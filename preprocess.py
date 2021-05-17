import nibabel as nib
import SimpleITK as itk
import numpy as np
import collections
import time
from nibabel.viewers import OrthoSlicer3D

def preprocess(filename):
	#读取文件
	open_filename = 'data/imagesTr/'+ filename +'.nii.gz'
	save_filename = 'data/imagesTr_Processed/'+ filename +'_Processed'+'.nii.gz'
	raw_img = nib.load(open_filename)
	raw_img_fdata = raw_img.get_fdata()
	data = np.transpose(raw_img_fdata, [2, 1, 0]) #转置
	(z, y, x) = data.shape

	#寻找截断集合
	reshape_data = data.ravel()
	counter = collections.Counter(reshape_data)
	#counter_list = list(counter.elements())
	num_X = round(0.05 * len(counter))
	num_Y = round(0.95 * len(counter))
	delete_list = []
	for i in range(len(counter)):
		if i < num_X:
			delete_list.append(counter[i])
		elif i > num_Y:
			delete_list.append(counter[i])

	# maxnum = counter[num_Y]
	# minnum = counter[num_X]

	# print(minnum)
	# print(maxnum)
	# min_list = counter_list[:num_X]
	# max_list = counter_list[num_Y:]

	#截断
	print('start:')
	tot = x * y * z
	count = 0
	for i in range(z):
	    for j in range(x):
	        for k in range(y):
	            value = data[i,j,k]
	            # if value < minnum or value > maxnum:
	            # 	data[i,j,k] = 0
	            if value in delete_list:
	            	data[i,j,k] = 0
	            count+=1
	            if count % 100000 == 0:
	            	print("当前进度：{}%".format(count/tot*100))

    #保存        
	img = itk.GetImageFromArray(data)
	itk.WriteImage(img, save_filename)
	print(filename + " 处理完成")



if __name__ == '__main__':
	time_start=time.time()
	for i in range(131):
		preprocess('liver_{}'.format(i))
		print('总进度：{}%'.format(i/130*100))
	time_end=time.time()
	print('totally cost (min):',(time_end-time_start)/60)