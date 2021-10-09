# -*— coding: utf-8 -*-
# @Time : 2021/6/15 10:09
# @Author : FYK

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from load_data import load_data, CovidDataset
from model import Linearmodel
from others import init_seed


cfg = {
    "data_root": "data",
    "save_path": "model_file",
    "num_epochs": 400,
    "batch_size": 512,
    "lr": 0.001,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

init_seed(0, cfg["device"])

train_data, train_label, val_data, val_label, test_data = load_data(cfg["data_root"])

train_set = CovidDataset(train_data, train_label)
times = int(len(train_set) / cfg["batch_size"]) + 1
val_set = CovidDataset(val_data, val_label)
test_set = CovidDataset(test_data)

train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
print("Data load successful!\n")

net = Linearmodel(in_dim=93).to(cfg["device"])

opt_Adam = optim.Adam(net.parameters(), lr=cfg["lr"])
criterion = nn.MSELoss(reduction="mean")


def train(train_dataset, model):
    model.train()
    loss = []
    for i, data in enumerate(train_dataset):
        data_x, data_y = data
        data_x, data_y = data_x.to(cfg["device"]), data_y.to(cfg["device"])
        opt_Adam.zero_grad()
        outputs = model(data_x)

        mse_loss = criterion(outputs, data_y)
        mse_loss.backward()
        opt_Adam.step()

        print("batch: {:2d}/{:2d}   train_loss: {:.4f}".format(i, times, mse_loss.detach().cpu().item()))
        loss.append(mse_loss.detach().cpu().item())

    return loss


def val(val_dataset, model):
    model.eval()
    loss = 0

    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            data_x, data_y = data
            data_x, data_y = data_x.to(cfg["device"]), data_y.to(cfg["device"])
            outputs = model(data_x)
            mse_loss = criterion(outputs, data_y)

            loss += mse_loss.detach().cpu().item() * len(data)
        loss /= len(val_dataset.dataset)
        return loss


def plt_loss(train_loss, val_loss):
    x1 = [i for i in range(len(train_loss))]
    x2 = [i for i in range(len(val_loss))]
    plt.subplot(1, 2, 1)
    plt.title('Train Loss')
    plt.plot(x1, train_loss, color='blue')
    plt.ylim(0.0, 10.)

    plt.subplot(1, 2, 2)
    plt.title('Val Loss')
    plt.plot(x2, val_loss, color='red')
    plt.ylim(0.0, 1.)
    plt.show()


def plot_pred(model, model_dir, device):
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    predicts, targets = [], []
    for i, data in enumerate(val_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            predicts.append(pred.detach().cpu())
            targets.append(y.detach().cpu())
    predicts = torch.cat(predicts, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, predicts, c='r', marker='.', alpha=0.5)
    plt.plot([-0.2, 35], [-0.2, 35], c='b')
    plt.xlim(-0.2, 35)
    plt.ylim(-0.2, 35)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()


if __name__ == "__main__":
    # train_loss = []
    # val_loss = []
    # for epoch in range(cfg["num_epochs"] + 1):
    #     print("************epoch-{:3d}************".format(epoch+1))
    #     loss = train(train_loader, net)
    #     train_loss.extend(loss)
    #
    #     print("\n***************Val***************")
    #     loss = val(val_loader, net)
    #     val_loss.append(loss)
    #     print("val_loss: {:.4f}".format(loss))
    #     print("***************Val***************\n")
    #
    #     if (epoch + 1) % 20 == 0:
    #         model_path = "epoch{}_loss{:.2f}.pth".format((epoch+1), loss)
    #         torch.save(net.state_dict(), os.path.join(cfg["save_path"], model_path))
    #
    # plt_loss(train_loss, val_loss)

    plot_pred(net, "model_file/epoch360_loss0.01.pth", cfg["device"])







