import json
import os
import random

import matplotlib
import nibabel
import nibabel as nib
import numpy as np
import scipy
import torch
from nibabel.viewers import OrthoSlicer3D
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy import ndimage
import SimpleITK as sitk


# matplotlib.use('TkAgg')


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, type, transform=None, do_resize=False, use_aug=True, auglist=['Lr', 'Ud'], read2D_image=True):
        self.do_resize = do_resize
        self.use_aug = use_aug
        self.auglist = auglist
        self.transform = transform
        self.read2D_image = read2D_image
        self.index_list = list(map(str, [index for index in range(0, 130)]))
        
        if read2D_image:
            self.train_image_path = os.listdir('data/imagesTr_2d')
            self.train_label_path = os.listdir('data/labelsTr_2d')
            random.shuffle(self.train_image_path)
            random.shuffle(self.train_label_path)
            divide_point = int(len(self.train_image_path) * 0.8)
            if type == "train":
                self.train_image_path = self.train_image_path[:divide_point]
                self.train_label_path = self.train_label_path[:divide_point]
                self.train_path = [{'image': self.train_image_path[index],
                                    'label': self.train_label_path[index]}
                                    for index in range(len(self.train_image_path))]
            else:
                self.train_image_path = self.train_image_path[divide_point:]
                self.train_label_path = self.train_label_path[divide_point:]
                self.train_path = [{'image': self.train_image_path[index],
                                    'label': self.train_label_path[index]}
                                    for index in range(len(self.train_image_path))]

        else:
            divide_point = int(len(self.index_list) * 0.8)
            random.shuffle(self.index_list)  # in place shuffle
            if type == "train":
                train_index = self.index_list[:divide_point]
                self.train_path = [
                    {'image': 'data/imagesTr_Cut/liver_{}_Cut.nii.gz'.format(index),
                        'label': 'data/labelsTr_Cut/liver_{}_Labels_Cut.nii.gz'.format(index)}
                    for index in train_index]
                if self.use_aug:
                    for augmethod in self.auglist:
                        self.train_path.extend([
                            {'image': f'data/imagesTr_{augmethod}/liver_{index}_{augmethod}.nii.gz',
                                'label': f'data/labelsTr_{augmethod}/liver_{index}_Labels_{augmethod}.nii.gz'}
                            for index in train_index])
                random.shuffle(self.train_path)  # in place shuffle
            else:
                train_index = self.index_list[divide_point:]
                self.train_path = [
                    {'image': 'data/imagesTr_Cut/liver_{}_Cut.nii.gz'.format(index),
                        'label': 'data/labelsTr_Cut/liver_{}_Labels_Cut.nii.gz'.format(index)}
                    for index in train_index]
                random.shuffle(self.train_path)  # in place shuffle

    def shuffle(self):
        random.shuffle(self.train_path)

    def __getitem__(self, index):
        img_path = self.train_path[index]["image"]
        label_path = self.train_path[index]["label"]

        if self.read2D_image:
            img = np.load(img_path) # type:
            label = np.load(label_path)
        else:
            if self.do_resize:
                img, label = resize(img_path, label_path)
            else:
                img = nib.load(img_path).get_fdata(dtype=np.float32)
                label = nib.load(label_path).get_fdata(dtype=np.float32)

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

            if not self.read2D_image:
                x, y, z = img.shape
                img = img.reshape(-1, x, y, z)
                label = label.reshape(-1, x, y, z)
                # img /= 300
            # else:
            #     x, y = img.shape
            #     img = img.reshape(-1, x, y)
            #     label = label.reshape(-1, x, y)

        return img, label  # 1 * 128 * 64 * 64

    def __len__(self):
        return len(self.train_path)


def resize(img_path, label_path):
    expand_slice = 0  # 轴向外侧扩张的slice数量

    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_array = sitk.GetArrayFromImage(img)
    label = sitk.ReadImage(label_path, sitk.sitkFloat32)
    label_array = sitk.GetArrayFromImage(label)
    image_divide = True  # 根据label进行目标切分
    image_show = False  # 展示分割后图像
    image_zoom = False  # 缩放图像
    image_save = True  # 保存numpy

    if image_divide:
        # 找到肝脏区域开始和结束的slice，并各向外扩张
        z = np.any(label_array, axis=(1, 2))
        start_slice, end_slice = np.where(z)[0][[0, -1]]

        # 两个方向上各扩张个slice
        if start_slice - expand_slice < 0:
            start_slice = 0
        else:
            start_slice -= expand_slice

        if end_slice + expand_slice >= label_array.shape[0]:
            end_slice = label_array.shape[0] - 1
        else:
            end_slice += expand_slice
        print("Cut out range:", str(start_slice) + '--' + str(end_slice))

        # # 截取保留区域
        img_array = img_array[start_slice:end_slice + 1, :, :]
        label_array = label_array[start_slice:end_slice + 1, :, :]
    if image_zoom:
        # # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
        x, y, z = img_array.shape
        xy_down_scale = 64 / y
        slice_down_scale = x / 128
        img_array = ndimage.zoom(img_array,
                                 (img.GetSpacing()[-1] / slice_down_scale, xy_down_scale, xy_down_scale),
                                 order=3)
        label_array = ndimage.zoom(label_array,
                                   (img.GetSpacing()[-1] / slice_down_scale, xy_down_scale, xy_down_scale),
                                   order=0)

    if image_show:
        save_filename = 'output.nii.gz'
        img = sitk.GetImageFromArray(img_array)
        sitk.WriteImage(img, save_filename)
        img = nib.load(save_filename)
        OrthoSlicer3D(img.dataobj).show()
    print("Preprocessed shape:", img_array.shape, label_array.shape)

    if image_save:
        for i in range(len(img_array)):
            np.save(f'data/imagesTr_Numpy/imagesTr_numpy_{i}.npy', img_array[i])
            np.save(f'data/labelsTr_Numpy/labelsTr_numpy_{i}.npy', label_array[i])
            print('完成{:.2f}%'.format(i / len(img_array) * 100))
    return img_array, label_array


def load_data(batch_size=8, do_resize=False, use_aug=True, auglist=['Lr', 'Ud'], read2D_image=True):
    random.seed(0)
    train_dataset = DealDataset("train", transform=transforms.ToTensor(), do_resize=do_resize,
                                use_aug=use_aug, auglist=auglist,
                                read2D_image=read2D_image)
    test_dataset = DealDataset("test", transform=transforms.ToTensor(), do_resize=do_resize,
                               use_aug=use_aug, auglist=auglist,
                               read2D_image=read2D_image)
    # 载入数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader


if __name__ == '__main__':
    # resize('data/imagesTr/liver_3.nii.gz',
    #        'data/labelsTr/liver_3.nii.gz')
    x = np.load(f'data/imagesTr_Numpy/imagesTr_numpy_{0}.npy')
    y = os.listdir('data/imagesTr_Numpy')
    random.shuffle(y)
    print(y)
    # train_loader, test_loader = load_data()