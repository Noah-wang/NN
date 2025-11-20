from torch import nn
import torch


class CIFAR10Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=64 * 4 * 4, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.linear1(x)

        x = self.linear2(x)
        x = self.softmax(x)

        return x