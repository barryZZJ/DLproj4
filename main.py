import torch
import os

from torch import nn


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


def test(model, device, test_loader):
    model.to(device)
    model.eval()
    acc = loss = 0
    total = 0

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

    acc /= total
    loss /= total
    return acc, loss



if __name__ == "__main__":
