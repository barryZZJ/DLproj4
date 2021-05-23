import json

import nibabel as nib
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, type, transform=None, use_aug=True, DEBUG=False):
        dataset_path = json.load(open("data/dataset.json"))["training"]
        self.transform = transform

        divide_point = int(len(dataset_path) * 0.8)
        if DEBUG:
            divide_point = 1

        if type == "train":
            self.train_path = dataset_path[:divide_point]
        else:
            if DEBUG:
                self.train_path = dataset_path[:divide_point]
            else:
                self.train_path = dataset_path[divide_point:]
        self.use_aug = use_aug
        self.DEBUG = DEBUG

    def __getitem__(self, index):
        img_path = "data/" + self.train_path[index]["image"]
        label_path = "data/" + self.train_path[index]["label"]
        if self.DEBUG:
            img_path = img_path.replace("imagesTr", "imagesTr_Processed").replace(".nii.gz", "_Processed.nii.gz")
            label_path = label_path.replace("labelsTr", "labelsTr_Processed").replace(".nii.gz", "_Labels_Processed.nii.gz")
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
            width, height, queue = img.shape
            padding = torch.zeros(990 - width, 512, 512)
            img = torch.cat([img, padding], 0) / 300
            label = torch.cat([label, padding], 0)
        return img, label  # 990 * 512 * 512

    def __len__(self):
        return len(self.train_path)


def load_data(batch_size=8, use_aug=True, DEBUG=False):
    train_dataset = DealDataset("train", transform=transforms.ToTensor(), use_aug=use_aug, DEBUG=DEBUG)
    test_dataset = DealDataset("test", transform=transforms.ToTensor(), use_aug=use_aug, DEBUG=DEBUG)
    # 载入数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader
