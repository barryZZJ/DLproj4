import json
from random import shuffle

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

matplotlib.use('TkAgg')


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, type, transform=None, use_cut=True, use_resize=True, DEBUG=False):
        self.use_resize = use_resize
        self.use_cut = use_cut
        self.DEBUG = DEBUG
        self.transform = transform
        self.index_list = list(map(str, [index for index in range(0, 130)]))
        if DEBUG:
            divide_point = 1
            train_index = self.index_list[:divide_point]
            self.train_path = [
                {'image': 'data/imagesTr_Cut/liver_num_Cut.nii.gz'.replace("num", index),
                 'label': 'data/labelsTr_Cut/liver_num_Labels_Cut.nii.gz'.replace("num", index)}
                for index in train_index]
        else:
            shuffle(self.index_list)
            divide_point = int(len(self.index_list) * 0.8)
            if type == "train":
                train_index = self.index_list[:divide_point]
                self.train_path = [
                    {'image': 'data/imagesTr_Cut/liver_num_Cut.nii.gz'.replace("num", index),
                     'label': 'data/labelsTr_Cut/liver_num_Labels_Cut.nii.gz'.replace("num", index)}
                    for index in train_index]
                self.train_path += [
                    {'image': 'data/imagesTr_Lr/liver_num_Lr.nii.gz'.replace("num", index),
                     'label': 'data/labelsTr_Lr/liver_num_Labels_Lr.nii.gz'.replace("num", index)}
                    for index in train_index]
                self.train_path += [
                    {'image': 'data/imagesTr_Ud/liver_num_Ud.nii.gz'.replace("num", index),
                     'label': 'data/labelsTr_Ud/liver_num_Labels_Ud.nii.gz'.replace("num", index)}
                    for index in train_index]

                shuffle(self.train_path)

            else:
                train_index = self.index_list[divide_point:]
                self.train_path = [
                    {'image': 'data/imagesTr_Cut/liver_num_Cut.nii.gz'.replace("num", index),
                     'label': 'data/labelsTr_Cut/liver_num_Labels_Cut.nii.gz'.replace("num", index)}
                    for index in train_index]

    def __getitem__(self, index):
        img_path = self.train_path[index]["image"]
        label_path = self.train_path[index]["label"]
        if self.use_resize:
            img, label = resize(img_path, label_path)
        else:
            img = nib.load(img_path).get_fdata(dtype=np.float32)
            label = nib.load(label_path).get_fdata(dtype=np.float32)

        if self.transform is not None:
            img = self.transform(img)  # type: torch.Tensor
            label = self.transform(label)
            x, y, z = img.shape
            img = img.reshape(-1, x, y, z)
            label = label.reshape(-1, x, y, z)
            img /= 300

        return img, label  # 1 * 128 * 64 * 64

    def __len__(self):
        return len(self.train_path)


def resize(img_path, label_path):
    expand_slice = 20  # 轴向外侧扩张的slice数量

    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_array = sitk.GetArrayFromImage(img)
    label = sitk.ReadImage(label_path, sitk.sitkFloat32)
    label_array = sitk.GetArrayFromImage(label)
    image_divide = False  # 根据label进行目标切分
    image_show = False  # 展示分割后图像

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
    return img_array, label_array


def load_data(batch_size=8, use_cut=True, use_resize=False, DEBUG=False):
    train_dataset = DealDataset("train", transform=transforms.ToTensor(), use_cut=use_cut, use_resize=use_resize,
                                DEBUG=DEBUG)
    test_dataset = DealDataset("test", transform=transforms.ToTensor(), use_cut=use_cut, use_resize=use_resize,
                               DEBUG=DEBUG)
    # 载入数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader


if __name__ == '__main__':
    # resize('data/imagesTr_Processed/liver_3_Processed.nii.gz',
    #        'data/labelsTr_Processed/liver_3_Labels_Processed.nii.gz')
    train_loader, test_loader = load_data()
