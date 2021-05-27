#%%

!pip install nibabel
!pip install SimpleITK

#%%

from obsmanip import OBS
bucket_name = 'zzjmnist'
base_path = 'DLproj4'
obs = OBS(bucket_name, base_path)

#%%

pyfiles = [filename for filename in obs.listdir('.') if filename.endswith('.py')]
for filename in pyfiles:
    obs.downloadFile(filename, filename)

#%%

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
def mkobsdir(path, obs:OBS):
    if not obs.exists(obs.abspath(path)):
        obs.mkdir(path)

def download(use_aug, auglist):
    mkdir('./data')
    obs.downloadFile('./data/dataset.json', './data/dataset.json')
    obs.downloadDir('./data/imagesTr_Cut', './data/imagesTr_Cut')
    obs.downloadDir('./data/labelsTr_Cut', './data/labelsTr_Cut')

    if use_aug:
        for augmethod in auglist:
            obs.downloadDir(f'./data/imagesTr_{augmethod}', f'./data/imagesTr_{augmethod}')
            obs.downloadDir(f'./data/labelsTr_{augmethod}', f'./data/labelsTr_{augmethod}')

    os.listdir('./data')

#%%

import torch
import os
import gc
from dice import *
from torch import nn
from dataloader import *
from modules import *

config = {"lr": 0.1,
          'momentum': 0.9,
          'batch_size': 8,
          'do_resize': False, # 我们上传的是已经处理好的图片，因此不再使用resize
          'use_aug': True,
          'auglist': ['Lr', 'Ud'], # 已经实现的增强方案
          "epochs": 200,
          'test_every': 10, # 每几个epoch测试一次
          'save_every': 10,
          'save_dir': './checkpoint',
          'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
          }
config_debug = {"lr": 0.1,
                'momentum': 0.9,
                'batch_size': 8,
                'do_resize': False, # 我们上传的是已经处理好的图片，因此不再使用resize
                'use_aug': False,
                'auglist': ['Lr', 'Ud'], # 已经实现的增强方案
                "epochs": 200,
                'test_every': 10, # 每几个epoch测试一次
                'save_every': 10,
                'save_dir': './checkpoint',
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                }
DEBUG = False
if DEBUG:
    config = config_debug

config['tostr'] = lambda : f"lr{config['lr']}_mom{config['momentum']}_bs{config['batch_size']}" + (f"_aug{''.join(config['auglist'])}" if config['use_aug'] else '')

download(config['use_aug'], config['auglist'])

#%%

def load_checkpoint_if_exists(model, save_dir, obs:OBS):
    save_dir = os.path.join(save_dir, config['tostr']())
    if obs.exists(obs.abspath(save_dir)):
        files = [int(filename[:-4]) for filename in obs.listdir(save_dir) if filename.endswith('.pth')]
    else:
        print(obs.pre(obs.abspath(save_dir)), "does not exists")
        files = []

    if files:
        max_file = f"{max(files)}.pth"
        print("load", save_dir, max_file)
        model.load_state_dict(torch.load(os.path.join(save_dir, max_file)))
        return max(files)
    print('load checkpoint fail')
    return 0

def save_model(model, epoch, save_dir, obs:OBS):
    mkdir(save_dir)
    mkobsdir(save_dir, obs)
    save_dir = os.path.join(save_dir, config['tostr']())
    mkdir(save_dir)
    mkobsdir(save_dir, obs)
    file = os.path.join(save_dir, f'{epoch}.pth')

    torch.save(model.state_dict(), file)
    obs.uploadFile(file, file)
    print("upload to", obs.pre(obs.abspath(file)))

def train(model, device, train_loader, optimizer):
    model = model.to(device)
    loss_save = 0
    dice_save = 0
    total = 0

    criterion = nn.BCEWithLogitsLoss().to(device)
    dice = SoftDiceLoss()
    dice.eval()

    for batch_idx, (x, labels) in enumerate(train_loader):
        gc.collect()
        x, labels = x.to(device), labels.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, labels)

        loss_save += loss.item() * labels.size(0)
        loss.backward()
        total += labels.size(0)

        dice_save += dice.forward(y_pred, labels)

        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    gc.collect()

    loss_save /= total
    dice_save /= total
    return loss_save, dice_save

def pred(model, device, test_loader):
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch_idx, (x, labels) = next(test_loader)
        x, labels = x.to(device), labels.to(device)
        y_pred = model(x)
    return y_pred


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    loss_save = 0
    dice_save = 0
    total = 0

    criterion = nn.BCEWithLogitsLoss().to(device)
    dice = SoftDiceLoss()
    dice.eval()

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            gc.collect()
            x, labels = x.to(device), labels.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, labels)

            loss_save += loss.item() * labels.size(0)
            total += labels.size(0)

            dice_save += dice.forward(y_pred, labels)
            torch.cuda.empty_cache()

    gc.collect()

    loss_save /= total
    dice_save /= total
    return loss_save, dice_save


if __name__ == "__main__":

    # device
    device = config['device']
    # load data
    train_loader, test_loader = load_data(config['batch_size'], config['do_resize'], config['use_aug'], config['auglist'], DEBUG=DEBUG)

    model = UNet(n_channels=1, n_classes=1, bilinear=False) # TODO bilinear?
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=0.001,
    #                              betas=(0.9, 0.999),
    #                              eps=1e-08,
    #                              weight_decay=0,
    #                              amsgrad=False)

    start_epoch = load_checkpoint_if_exists(model, config['save_dir'], obs)

    train_losses = []
    train_dices = []
    test_losses = []
    test_dices = []

    for epoch in range(1+start_epoch, config['epochs']+1):
        print("Training epoch:\t", epoch)
        loss, dice = train(model, device, train_loader, optimizer)
        print(f'### Train ### Epoch: {epoch} \n'
              f'loss: {loss}\tdice: {dice}')
        train_losses.append(','.join([str(epoch), str(loss)]))
        train_losses.append('\n')
        train_dices.append(','.join([str(epoch), str(dice)]))
        train_dices.append('\n')

        if epoch % config['test_every'] == 0:
            loss, dice = test(model, device, test_loader)
            save_model(model, epoch, config['save_dir'], obs)
            test_losses.append(','.join([str(epoch), str(loss)]))
            test_losses.append('\n')
            test_dices.append(','.join([str(epoch), str(dice)]))
            test_dices.append('\n')
            print(f'### Test ### Epoch: {epoch} \n'
                  f'loss: {loss}\tdice: {dice}')

    with obs.open('./train_loss.csv', 'w') as f:
        f.writelines(train_losses)
    with obs.open('./train_dice.csv', 'w') as f:
        f.writelines(train_dices)
    with obs.open('./test_loss.csv', 'w') as f:
        f.writelines(test_losses)
    with obs.open('./test_dice.csv', 'w') as f:
        f.writelines(test_dices)
