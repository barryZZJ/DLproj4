import json

import nibabel as nib
import numpy as np
import torch
from nibabel.viewers import OrthoSlicer3D
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from scipy import ndimage
import SimpleITK as sitk


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, type, transform=None, use_cut=True, DEBUG=False):
        if DEBUG:
            dataset_path = json.load(open("data/dataset_debug.json"))["training"]
        else:
            dataset_path = json.load(open("data/dataset.json"))["training"]
        self.transform = transform

        divide_point = int(len(dataset_path) * 0.8)
        if type == "train":
            self.train_path = dataset_path[:divide_point]
        else:
            self.train_path = dataset_path[divide_point:]
        self.use_cut = use_cut
        self.DEBUG = DEBUG

    def __getitem__(self, index):
        img_path = "data/" + self.train_path[index]["image"].replace("./", "")
        label_path = "data/" + self.train_path[index]["label"].replace("./", "")
        if self.use_cut:
            img_path = img_path.replace("imagesTr", "imagesTr_Processed").replace(".nii.gz", "_Processed.nii.gz")
            label_path = label_path.replace("labelsTr", "labelsTr_Processed").replace(".nii.gz",
                                                                                      "_Labels_Processed.nii.gz")
        print(img_path)
        img, label = resize(img_path, label_path)

        if self.transform is not None:
            img = self.transform(img)  # type: torch.Tensor
            label = self.transform(label)
            print(img.shape)
            height, width, queue = img.shape
            print(width, height, queue)
            img = img.reshape(-1, height, width, queue)
            label = label.reshape(-1, height, width, queue)

            print(img.shape, label.shape)
            img /= 300
            padding = torch.zeros(1, height, 128 - width, queue)
            # print(padding.shape)
            img = torch.cat([img, padding], 2)
            label = torch.cat([label, padding], 2)
        print(img.shape, label.shape)
        return img, label  # 1 * 128 * 64 * 64

    def __len__(self):
        return len(self.train_path)


def resize(img_path, label_path):
    expand_slice = 10  # 轴向外侧扩张的slice数量
    min_size = 1  # 取样的slice数量
    xy_down_scale = 0.125
    slice_down_scale = 2.5

    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_array = sitk.GetArrayFromImage(img)
    label = sitk.ReadImage(label_path, sitk.sitkFloat32)
    label_array = sitk.GetArrayFromImage(label)

    print("Ori shape:", img_array.shape, label_array.shape)
    # # 降采样，（对x和y轴进行降采样，slice轴的spacing归一化到slice_down_scale）
    print(img.GetSpacing()[-1] / slice_down_scale)
    img_array = ndimage.zoom(img_array,
                             (img.GetSpacing()[-1] / slice_down_scale, xy_down_scale, xy_down_scale),
                             order=3)
    label_array = ndimage.zoom(label_array,
                               (img.GetSpacing()[-1] / slice_down_scale, xy_down_scale, xy_down_scale),
                               order=0)
    print(img_array.shape)
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
    # 如果这时候剩下的slice数量不足size，直接放弃，这样的数据很少
    # if end_slice - start_slice + 1 < min_size:
    #     print('Too little slice，give up the sample:', "ct_file")
    #     return None, None
    # # 截取保留区域
    img_array = img_array[start_slice:end_slice + 1, :, :]
    label_array = label_array[start_slice:end_slice + 1, :, :]
    print("Preprocessed shape:", img_array.shape, label_array.shape)
    return img_array, label_array

def load_data(batch_size=8, use_cut=True, DEBUG=False):
    train_dataset = DealDataset("train", transform=transforms.ToTensor(), use_cut=use_cut, DEBUG=DEBUG)
    test_dataset = DealDataset("test", transform=transforms.ToTensor(), use_cut=use_cut, DEBUG=DEBUG)
    # 载入数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    return train_loader, test_loader


if __name__ == '__main__':
    resize('data/imagesTr_Processed/liver_0_Processed.nii.gz',
           'data/labelsTr_Processed/liver_0_Labels_Processed.nii.gz')
