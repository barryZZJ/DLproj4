
#%%

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

#%%

import torch
import os
import gc
from dice import *
from torch import nn
from dataloader import *
from models import UNet, UNetv2, UNet3d, ResUNet, R2UNet

config = {
    'optimizer': 'adam',
    'sgd':{'lr':0.1,
           'momentum':0.9},
    'adam':{'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps':1e-08,
            'weight_decay': 0.001,
            'amsgrad': True},

    'batch_size': 8,
    'do_resize': False, # 我们上传的是已经处理好的图片，因此不再使用resize
    'use_aug': False,
    'auglist': ['Lr', 'Ud', 'Rot', 'Turn', 'Shift', 'Zoom', 'Gamma'], # 已经实现的增强方案
    'read2D_image': True, # 使用已切片的2d图像
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
                }
DEBUG = False
if DEBUG:
    for key,val in config_debug.items():
        config[key]=val

config['tostr'] = lambda : f"opt{config['optimizer']}_{'_'.join(['{}{}'.format(key, val) for key, val in config[config['optimizer']].items()])}_bs{config['batch_size']}" + (f"_aug{''.join(sorted(config['auglist']))}" if config['use_aug'] else '')


#%%

# download(config['use_aug'], config['auglist'])

#%%

def load_checkpoint_if_exists(model, save_dir):
    save_dir = os.path.join(save_dir, config['tostr']())
    if os.path.exists(save_dir):
        files = [int(filename[:-4]) for filename in os.listdir(save_dir) if filename.endswith('.pth')]
    else:
        print(save_dir, "does not exists")
        files = []

    if files:
        max_file = os.path.join(save_dir, f"{max(files)}.pth")
        print("load", max_file)
        model.load_state_dict(torch.load(max_file))
        return max(files)
    print('load checkpoint fail')
    return 0

def save_loss(filename, losses, save_dir):
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, config['tostr']())
    mkdir(save_dir)
    file = os.path.join(save_dir, f'{filename}.csv')

    with open(file, 'a') as f:
        f.writelines(losses)


def save_model(model, epoch, save_dir):
    mkdir(save_dir)
    save_dir = os.path.join(save_dir, config['tostr']())
    mkdir(save_dir)
    file = os.path.join(save_dir, f'{epoch}.pth')

    torch.save(model.state_dict(), file)

def train(model, device, train_loader, optimizer):
    model = model.to(device)
    model.train()
    loss_save = 0
    dice_save = 0
    total = 0

    criterion = nn.BCEWithLogitsLoss().to(device)
    # criterion = DiceLoss().to(device)

    for batch_idx, (x, labels) in enumerate(train_loader):
        gc.collect()
        x, labels = x.to(device), labels.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, labels)

        loss_save += loss.item() * labels.size(0)
        loss.backward()
        total += labels.size(0)

        dice_save += dice_coef(y_pred, labels)

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
    # criterion = DiceLoss().to(device)

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            gc.collect()
            x, labels = x.to(device), labels.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, labels)

            loss_save += loss.item() * labels.size(0)
            total += labels.size(0)

            dice_save += dice_coef(y_pred, labels)
            torch.cuda.empty_cache()

    gc.collect()

    loss_save /= total
    dice_save /= total
    return loss_save, dice_save

#%%

if __name__ == "__main__":

    # device
    device = config['device']
    # load data
    train_loader, test_loader = load_data(config['batch_size'], config['do_resize'], config['use_aug'], config['auglist'], config['read2D_image'])

    # model = UNet(n_channels=1, n_classes=1, bilinear=False) # TODO bilinear?
    model = UNetv2(img_ch=1, output_ch=1)

    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **config['sgd'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), **config['adam'])

    start_epoch = load_checkpoint_if_exists(model, config['save_dir'])

    test_losses = []
    test_dices = []

    for epoch in range(1+start_epoch, config['epochs']+1):
        print(f'### Train ### Epoch: {epoch}')
        loss, dice = train(model, device, train_loader, optimizer)
        train_loader.dataset.shuffle() # 每个epoch shuffle
        print(f'loss: {loss}\tdice: {dice}')
        save_loss('train_loss', [','.join([str(epoch), str(loss)]), '\n'], config['save_dir'])
        # save_loss('train_dice', [','.join([str(epoch), str(dice)]), '\n'], config['save_dir'])

        if epoch % config['test_every'] == 0:
            tloss, tdice = test(model, device, test_loader)
            test_losses.append(','.join([str(epoch), str(tloss)]))
            test_losses.append('\n')
            test_dices.append(','.join([str(epoch), str(tdice)]))
            test_dices.append('\n')
            print(f'### Test ### Epoch: {epoch}')
            print(f'loss: {tloss}\tdice: {tdice}')

        if epoch % config['save_every'] == 0:
            save_model(model, epoch, config['save_dir'])
            # save_loss('test_loss', test_losses, config['save_dir'])
            save_loss('test_dice', test_dices, config['save_dir'])
            test_losses = []
            test_dices = []
