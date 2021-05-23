import torch
import os
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

config_debug = {"learning_rate": 1e-3,
                'batch_size': 1,
                'use_cut': True,
                "epochs": 1,
                'test_every': 1, # 每几个epoch测试一次
                'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
                }

def train(model, device, train_loader, optimizer):
    model = model.to(device)
    loss_save = 0
    for i_batch, (img, label) in enumerate(train_loader):
        img = img.to(device)
        label = label.to(device)
        y_pred = model(img)

        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(y_pred, label)
        loss_save = loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
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
    acc = loss = 0
    total = 0
    dice_loss = 0

    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(test_loader):
            x, labels = x.to(device), labels.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, labels.long())

            _, predicted = torch.max(y_pred, -1)
            correct = predicted.eq(labels).sum()

            acc += correct.item()
            loss += loss.item() * labels.size(0)
            total += labels.size(0)

            dice = SoftDiceLoss(num_classes)
            dice_loss += SoftDiceLoss(y_pred, labels.long())

    acc /= total
    loss /= total
    dice /= total
    return acc, loss, dice_loss


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        config = config_debug

    # device
    device = config['device']
    # load data
    train_loader, test_loader = load_data(batch_size=config['batch_size'], use_cut=config['use_cut'], DEBUG=DEBUG)

    model = UNet(n_channels=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=0.001,
    #                              betas=(0.9, 0.999),
    #                              eps=1e-08,
    #                              weight_decay=0,
    #                              amsgrad=False)

    for epoch in range(1, config['epochs']+1):
        train(model, device, train_loader, optimizer)
        if epoch % config['test_every'] == 0:
            acc, loss, dice_loss = test(model, device, test_loader, 2)
            print(f'### Epoch: {epoch} \n'
                  f'acc: {acc}\tdice loss: {loss}\tdice_loss: {dice_loss}')
