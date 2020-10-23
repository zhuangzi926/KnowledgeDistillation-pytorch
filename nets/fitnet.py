""" Detailed descriptions about arch of FitNets:
        https://arxiv.org/abs/1412.6550
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.maxout import ListModule
# from maxout import ListModule


class FitNetMNIST(nn.Module):
    """ 6-conv deep and thin FitNet. """

    def __init__(self, output_size=10, num_units=2):
        super(FitNetMNIST, self).__init__()
        self.conv1_list = ListModule(self, "conv1_")
        self.conv2_list = ListModule(self, "conv2_")
        self.conv3_list = ListModule(self, "conv3_")
        self.conv4_list = ListModule(self, "conv4_")
        self.conv5_list = ListModule(self, "conv5_")
        self.conv6_list = ListModule(self, "conv6_")
        self.pool1 = nn.MaxPool2d(4, 2)
        self.pool2 = nn.MaxPool2d(4, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
        self.fc = nn.Linear(192, out_features=output_size)

        for _ in range(num_units):
            self.conv1_list.append(nn.Conv2d(1, 16, 3, 1, padding=1))
            self.conv2_list.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.conv3_list.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.conv4_list.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.conv5_list.append(nn.Conv2d(16, 12, 3, 1, padding=1))
            self.conv6_list.append(nn.Conv2d(12, 12, 3, 1, padding=1))

        self.activation = {}
        self.pool2.register_forward_hook(self.get_activation("guided_layer"))

    def forward(self, x):
        x = self.maxout(x, self.conv1_list)
        x = self.pool1(self.maxout(x, self.conv2_list))
        x = self.maxout(x, self.conv3_list)
        x = self.pool2(self.maxout(x, self.conv4_list))
        x = self.maxout(x, self.conv5_list)
        x = self.pool3(self.maxout(x, self.conv6_list))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = self.fc(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # return F.softmax(x, dim=1)
        return x

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


class FitNet1CIFAR(nn.Module):
    """ 9-conv deep and thin FitNet. """

    def __init__(self, output_size=10, num_units=2):
        super(FitNet1CIFAR, self).__init__()
        self.conv1_list = ListModule(self, "conv1_")
        self.conv2_list = ListModule(self, "conv2_")
        self.conv3_list = ListModule(self, "conv3_")
        self.conv4_list = ListModule(self, "conv4_")
        self.conv5_list = ListModule(self, "conv5_")
        self.conv6_list = ListModule(self, "conv6_")
        self.conv7_list = ListModule(self, "conv7_")
        self.conv8_list = ListModule(self, "conv8_")
        self.conv9_list = ListModule(self, "conv9_")
        # self.pool1 = nn.MaxPool2d(1, 1)
        # self.pool2 = nn.MaxPool2d(1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.pool4 = nn.MaxPool2d(1, 1)
        # self.pool5 = nn.MaxPool2d(1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # self.pool7 = nn.MaxPool2d(1, 1)
        # self.pool8 = nn.MaxPool2d(1, 1)
        self.pool3 = nn.MaxPool2d(8, 1)
        self.fc = nn.Linear(64, out_features=output_size)

        for _ in range(num_units):
            self.conv1_list.append(nn.Conv2d(3, 16, 3, 1, padding=1))
            self.conv2_list.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.conv3_list.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.conv4_list.append(nn.Conv2d(16, 32, 3, 1, padding=1))
            self.conv5_list.append(nn.Conv2d(32, 32, 3, 1, padding=1))
            self.conv6_list.append(nn.Conv2d(32, 32, 3, 1, padding=1))
            self.conv7_list.append(nn.Conv2d(32, 48, 3, 1, padding=1))
            self.conv8_list.append(nn.Conv2d(48, 48, 3, 1, padding=1))
            self.conv9_list.append(nn.Conv2d(48, 64, 3, 1, padding=1))

        self.activation = {}
        self.pool2.register_forward_hook(self.get_activation("guided_layer"))

    def forward(self, x):
        x = self.maxout(x, self.conv1_list)
        x = self.maxout(x, self.conv2_list)
        x = self.pool1(self.maxout(x, self.conv3_list))
        x = self.maxout(x, self.conv4_list)
        x = self.maxout(x, self.conv5_list)
        x = self.pool2(self.maxout(x, self.conv6_list))
        x = self.maxout(x, self.conv7_list)
        x = self.maxout(x, self.conv8_list)
        x = self.pool3(self.maxout(x, self.conv9_list))
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = F.dropout(x, p=0.2, training=self.training)
        # return F.softmax(x, dim=1)
        return x

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model = FitNetMNIST().to("cuda")
    # model = FitNet1CIFAR().to("cuda")
    data = torch.arange(28*28*1, dtype=torch.float).view(1, 1, 28, 28).to("cuda")
    # data = torch.arange(32 * 32 * 3, dtype=torch.float).view(1, 3, 32, 32).to("cuda")
    model(data)
    print(model.activation["guided_layer"].shape)
