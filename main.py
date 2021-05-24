# #%%
#
# !pip install nibabel
# !pip install SimpleITK
#
# #%%
# import os
# from modelarts.session import Session
# session = Session()
# if not os.path.exists('data'):
#     os.mkdir('data')
# if not os.path.exists('data/imagesTr'):
#     os.mkdir('data/imagesTr')
# if not os.path.exists('data/labelsTr'):
#     os.mkdir('data/labelsTr')
# if not os.path.exists('data/imagesTr_Processed'):
#     os.mkdir('data/imagesTr_Processed')
# if not os.path.exists('data/labelsTr_Processed'):
#     os.mkdir('data/labelsTr_Processed')
#
# bucket_base = '/cqu-hjk/project/data/'
# # bucket_base = '/zzjmnist/DLproj4/data/'
#
# session.download_data(bucket_path=bucket_base + "dataset.json", path="./data/dataset.json")
# session.download_data(bucket_path=bucket_base + "dataset_debug.json", path="./data/dataset_debug.json")
# livers = 131
# for i in range(livers):
#     session.download_data(bucket_path=bucket_base + f"imagesTr/liver_{i}.nii.gz", path="./data/imagesTr/liver_{i}.nii.gz")
#     session.download_data(bucket_path=bucket_base + f"labelsTr/liver_{i}.nii.gz", path="./data/labelsTr/liver_{i}.nii.gz")
#     session.download_data(bucket_path=bucket_base + "imagesTr_Processed/liver_{i}_Processed.nii.gz", path="./data/imagesTr_Processed/liver_{i}_Processed.nii.gz")
#     session.download_data(bucket_path=bucket_base + "labelsTr_Processed/liver_{i}_Labels_Processed.nii.gz", path="./data/labelsTr_Processed/liver_{i}_Labels_Processed.nii.gz")

#%%
import torch
import os
import gc
from dice import *
from torch import nn
from dataloader import *
from modules import *

config = {"learning_rate": 1e-3,
          'batch_size': 8,
          'use_cut': True,
          "epochs": 200,
          'test_every': 10, # 每几个epoch测试一次
          'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
          }


def train(model, device, train_loader, optimizer):
    model = model.to(device)
    loss_save = 0
    criterion = nn.BCEWithLogitsLoss().to(device)
    for i_batch, (img, label) in enumerate(train_loader):
        gc.collect()
        img = img.to(device)
        label = label.to(device)
        y_pred = model(img)

        loss = criterion(y_pred, label)
        loss_save = loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    gc.collect()

    print("Train loss\t", loss_save)


def pred(model, device, test_loader):
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch_idx, (x, labels) = next(test_loader)
        x, labels = x.to(device), labels.to(device)
        y_pred = model(x)
    return y_pred


def test(model, device, test_loader, num_classes):
    model.to(device)
    model.eval()
    loss = 0
    total = 0
    dice_loss = 0

    criterion = nn.BCEWithLogitsLoss().to(device)

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            gc.collect()
            x, labels = x.to(device), labels.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, labels)

            loss += loss.item() * labels.size(0)
            total += labels.size(0)

            dice = SoftDiceLoss(num_classes)
            dice_loss += dice.forward(y_pred, labels)
    gc.collect()

    loss /= total
    dice_loss /= total
    return loss, dice_loss


if __name__ == "__main__":
    DEBUG = False
    config_debug = {"learning_rate": 1e-3,
                  'batch_size': 8,
                  'use_cut': False,
                  "epochs": 200,
                  'test_every': 10, # 每几个epoch测试一次
                  'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  }

    if DEBUG:
        config = config_debug

    # device
    device = config['device']
    # load data
    train_loader, test_loader = load_data(batch_size=config['batch_size'], use_cut=config['use_cut'], DEBUG=DEBUG)

    model = UNet(n_channels=1, n_classes=1, bilinear=False) # TODO bilinear?
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=0.001,
    #                              betas=(0.9, 0.999),
    #                              eps=1e-08,
    #                              weight_decay=0,
    #                              amsgrad=False)

    for epoch in range(1, config['epochs']+1):
        print("Training epoch:\t", epoch)
        train(model, device, train_loader, optimizer)
        if epoch % config['test_every'] == 0:
            loss, dice_loss = test(model, device, test_loader, 2)
            print(f'### Epoch: {epoch} \n'
                  f'loss: {loss}\tdice_loss: {dice_loss}')