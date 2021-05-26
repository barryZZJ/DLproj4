#%%
!pip install nibabel
!pip install SimpleITK
!pip install wandb
#%%
from obs import OBS
import os
bucket_name = 'zzjmnist'
base_path = 'DLproj4'
obs = OBS(bucket_name, base_path)
#%%
obs.downloadDir('./data', './data')
os.listdir('./data')
#%%
import torch
import os
import gc
from dice import *
from torch import nn
from dataloader import *
from modules import *
import wandb

config = {"lr": 0.1,
          'momentum': 0.9,
          'batch_size': 8,
          'use_cut': True,
          "epochs": 200,
          'test_every': 10, # 每几个epoch测试一次
          'save_every': 10,
          'save_dir': './checkpoint',
          'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
          }

def load_checkpoint_if_exists(model, save_dir, obs:OBS=None):
    if obs is not None:
        if obs.exists(obs.abspath(save_dir)):
            files = [int(filename[:-4]) for filename in obs.listdir(save_dir) if filename.endswith('.pth')]
        else:
            print(obs.pre(obs.abspath(save_dir)), "does not exists")
            files = []
    else:
        if os.path.exists(save_dir):
            files = [int(filename[:-4]) for filename in os.listdir(save_dir) if filename.endswith('.pth')]
        else:
            print((save_dir), "does not exists")
            files = []

    if files:
        max_file = f"{max(files)}.pth"
        print("load " + max_file)
        model.load_state_dict(torch.load(os.path.join(save_dir, max_file)))
        return max(files)
    print('load checkpoint fail')
    return 0

def save_model(model, epoch, save_dir, obs:OBS=None):
    file = f'{save_dir}/{epoch}.pth'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(model.state_dict(), file)
    if obs is not None:
        if not obs.exists(obs.abspath(save_dir)):
            obs.mkdir(save_dir)
        obs.uploadFile(file, file)
        print("upload to", obs.pre(obs.abspath(file)))
    print('save done')

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
    DEBUG = False
    config_debug = {"lr": 0.1,
                    'momentum': 0.9,
                    'batch_size': 8,
                    'use_cut': False,
                    "epochs": 200,
                    'test_every': 10, # 每几个epoch测试一次
                    'save_every': 10,
                    'save_dir': './checkpoint',
                    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    }
    if DEBUG:
        config = config_debug

    # https://wandb.ai/dlproj/unet
    wandb.init(project='unet', entity='dlproj', config=config)

    # device
    device = config['device']
    # load data
    train_loader, test_loader = load_data(batch_size=config['batch_size'], use_cut=config['use_cut'], DEBUG=DEBUG)

    model = UNet(n_channels=1, n_classes=1, bilinear=False) # TODO bilinear?
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=0.001,
    #                              betas=(0.9, 0.999),
    #                              eps=1e-08,
    #                              weight_decay=0,
    #                              amsgrad=False)

    start_epoch = load_checkpoint_if_exists(model, config['save_dir'], obs)

    for epoch in range(1+start_epoch, config['epochs']+1):
        print("Training epoch:\t", epoch)
        loss, dice = train(model, device, train_loader, optimizer)
        print(f'### Train ### Epoch: {epoch} \n'
              f'loss: {loss}\tdice: {dice}')
        wandb.log({f'train loss': loss, "epoch": epoch})
        wandb.log({f'train dice': dice, "epoch": epoch})

        if epoch % config['test_every'] == 0:
            loss, dice = test(model, device, test_loader)
            save_model(model, epoch, config['save_dir'], obs)
            wandb.log({f'test loss': loss, "epoch": epoch})
            wandb.log({f'test dice': dice, "epoch": epoch})
            print(f'### Test ### Epoch: {epoch} \n'
                  f'loss: {loss}\tdice: {dice}')