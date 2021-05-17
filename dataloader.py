import json

import nibabel as nib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import imageio
import matplotlib.pyplot as plt


class DealDataset(Dataset):
    """
        读取数据、初始化数据
    """

    def __init__(self, type, transform=None):
        datasetsPath = json.load(open("data/dataset.json"))["training"]
        self.transform = transform
        dividePoint = int(len(datasetsPath) * 0.8)
        if type == "train":
            self.trainPath = datasetsPath[dividePoint:]
        else:
            self.trainPath = datasetsPath[:dividePoint]

    def __getitem__(self, index):
        imgPath = "data/" + self.trainPath[index]["image"]
        imgPath = imgPath.replace("imagesTr", "imagesTr_Processed").replace(".nii.gz", "_Processed.nii.gz'")

        labelPath = "data/imagesTr_Processed/" + self.trainPath[index]["label"]
        labelPath = labelPath.replace("imagesTr", "imagesTr_Processed").replace(".nii.gz", "_Processed.nii.gz'")
        img = nib.load(imgPath).get_fdata()
        label = nib.load(labelPath).get_fdata()
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.trainPath)


def load_data():
    train_dataset = DealDataset("train", transform=transforms.ToTensor())
    test_dataset = DealDataset("test", transform=transforms.ToTensor())

    batchsize = 8
    # 载入数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=True)
    return train_loader, test_loader
