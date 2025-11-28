import copy
import time
from altair import Data
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data
import torchvision
from torchvision.datasets import FashionMNIST
from model import AlexNet
from torchvision import transforms


def test_data_process():
    test_data = FashionMNIST(
        root="./data",
        train=False,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Resize(size=227), torchvision.transforms.ToTensor()]
        ),
        download=True,
    )

    test_dataloader = data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=True, num_workers=0
    )
    return test_dataloader


test_dataloader = test_data_process()


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_correct = 0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            pre_lab = torch.argmax(output, dim=1)
            test_correct += torch.sum(pre_lab == test_data_y).item()
            test_num += test_data_x.size(0)

    test_acc = test_correct / test_num
    print("测试集上的准确率为: {:.4f}".format(test_acc))


if __name__ == "__main__":
    model = AlexNet()

    model.load_state_dict(torch.load("AlexNet/best_model.pth"))

    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = test_data_y.item()

            print("预测结果: ", classes[result], "------" "真实标签: ", classes[label])
