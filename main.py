import torch
import os
from dice import *
from torch import nn

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

            dice = SoftDiceLossV2(num_classes)
            dice_loss += SoftDiceLossV2(y_pred, labels.long())

    acc /= total
    loss /= total
    dice /= total
    return acc, loss, dice_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)

    for epoch in range(config.get('epochs')):
        pass



