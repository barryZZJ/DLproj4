import torch
import os
from dice import *
from torch import nn
from dataloader import *
from modules import *

config = {"learning_rate": 1e-3,
          "epochs": 10}


def train(model, device, train_loader, optimizer):
    model = model.to(device)
    loss_save = 0
    for i_batch, batch_data in enumerate(train_loader):
        img = batch_data[0].to(device)
        label = batch_data[1].to(device)
        y_pred = model(img)

        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(y_pred, label)
        loss_save = loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    print("Loss\t", loss_save)


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
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load data
    train_loader, test_loader = load_data()

    model = UNet(n_channels=1, n_classes=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=0.001,
    #                              betas=(0.9, 0.999),
    #                              eps=1e-08,
    #                              weight_decay=0,
    #                              amsgrad=False)

    for epoch in range(config.get('epochs')):
        train(model, device, train_loader, optimizer)
        if epoch % 1 == 0:
            acc, loss, dice_loss = test(model, device, test_loader, 2)
            print(f'### Epoch: {epoch} \n'
                  f'acc: {acc}\tloss: {loss}\tdice_loss: {dice_loss}')
