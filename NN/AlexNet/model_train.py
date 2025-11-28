import pandas as pd
import torch
from torch import nn
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
from model import AlexNet
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import copy
import time


# 数据处理
def train_val_data_process():
    # 下载数据集
    train_data = FashionMNIST(
        root="./data",
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(size=227), transforms.ToTensor()]
        ),
        download=True,
    )

    # 分开训练集和验证集
    train_data, val_data = data.random_split(
        train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))]
    )

    # 加载训练集
    train_dataloader = data.DataLoader(
        dataset=train_data, batch_size=32, shuffle=True, num_workers=2
    )

    # 加载验证集
    val_dataloader = data.DataLoader(
        dataset=val_data, batch_size=32, shuffle=False, num_workers=2
    )
    return train_dataloader, val_dataloader


# 模型训练
def train_model_process(model, train_dataloader, val_dataload, num_epochs):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    # 损失
    train_loss_all = []
    val_loss_all = []
    # 准确率
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        train_loss = 0.0
        train_corrects = 0

        val_loss = 0.0
        val_corrects = 0

        train_num = 0
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.train()

            outputs = model(b_x)

            pre_lab = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y).item()
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataload):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)

            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab == b_y).item()
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects / val_num)
        print(
            "train Loss: {:.4f} Acc: {:.4f}".format(
                train_loss / train_num, train_corrects / train_num
            )
        )
        print(
            "val   Loss: {:.4f} Acc: {:.4f}".format(
                val_loss / val_num, val_corrects / val_num
            )
        )

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print(
            "训练和验证耗费的时间: {:.0f}m {:.0f}s".format(
                time_use // 60, time_use % 60
            )
        )

    # 保存最优模型参数
    torch.save(best_model_wts, "AlexNet/best_model.pth")

    train_process = pd.DataFrame(
        data={
            "epoch": list(range(1, len(train_loss_all) + 1)),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        train_process["epoch"], train_process.train_loss_all, "bo-", label="train loss"
    )

    plt.plot(
        train_process["epoch"], train_process.val_loss_all, "ro-", label="val loss"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(
        train_process["epoch"],
        train_process.train_acc_all,
        "bo-",
        label="train accuracy",
    )

    plt.plot(
        train_process["epoch"], train_process.val_acc_all, "ro-", label="val accuracy"
    )
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model = AlexNet()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(
        model, train_dataloader, val_dataloader, num_epochs=20
    )
    matplot_acc_loss(train_process)
