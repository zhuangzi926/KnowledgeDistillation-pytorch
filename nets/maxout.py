""" See detailed analysis about maxout via links below: 
        https://github.com/Duncanswilson/maxout-pytorch/blob/master/maxout_pytorch.ipynb
        https://cs231n.github.io/neural-networks-1/

    Detailed descriptions about arch of MaxoutConv:
        https://github.com/paniabhisek/maxout/blob/master/maxout.json
        https://arxiv.org/abs/1412.6550
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ListModule(object):
    """A list-like object which includes all candidate rectifiers/convs of a single hidden layer for maxout operation."""

    def __init__(self, module, prefix, *args):
        """
        Args:
            module(nn.module): parent nn.module
            prefix(str): name prefix of list items
            args(list): list of nn.modules
        """
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError("Not a Module")
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError("Out of bound")
        return getattr(self.module, self.prefix + str(i))


class MaxoutConvMNIST(nn.Module):
    """ Model of 3-layer-ConvNet with maxout as activition function."""

    def __init__(self, output_size=10, num_units=2):
        super(MaxoutConvMNIST, self).__init__()
        self.conv1_list = ListModule(self, "conv1_")
        self.conv2_list = ListModule(self, "conv2_")
        self.conv3_list = ListModule(self, "conv3_")
        self.fc = nn.Linear(96, out_features=output_size)

        for _ in range(num_units):
            self.conv1_list.append(nn.Conv2d(1, 48, 7, 1, padding=3))
            self.conv2_list.append(nn.Conv2d(48, 48, 7, 1, padding=2))
            self.conv3_list.append(nn.Conv2d(48, 24, 5, 1, padding=2))

    def forward(self, x):
        x = F.max_pool2d(self.maxout(x, self.conv1_list), 4, stride=2)
        x = F.max_pool2d(self.maxout(x, self.conv2_list), 4, stride=2)
        x = F.max_pool2d(self.maxout(x, self.conv3_list), 2, stride=2)
        # print(x.shape)
        x = x.view(-1, 96)
        x = self.fc(x)
        x = F.dropout(x, training=self.training)
        return F.softmax(x, dim=1)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output

class MaxoutConvCIFAR(nn.Module):
    """ Model of 3-layer-ConvNet with maxout as activition function."""

    def __init__(self, output_size=10, conv_num_units=2, fc_num_units=5):
        super(MaxoutConvCIFAR, self).__init__()
        self.conv1_list = ListModule(self, "conv1_")
        self.conv2_list = ListModule(self, "conv2_")
        self.conv3_list = ListModule(self, "conv3_")
        self.conv1_bn = nn.BatchNorm2d(96)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.conv3_bn = nn.BatchNorm2d(192)
        self.fc1_list = ListModule(self, "fc1_")
        self.fc = nn.Linear(500, out_features=output_size)

        for _ in range(conv_num_units):
            self.conv1_list.append(nn.Conv2d(3, 96, 7, 1, padding=3))
            self.conv2_list.append(nn.Conv2d(96, 192, 7, 1, padding=2))
            self.conv3_list.append(nn.Conv2d(192, 192, 5, 1, padding=2))
        
        for _ in range(fc_num_units):
            self.fc1_list.append(nn.Linear(768, 500))

    def forward(self, x):
        x = F.max_pool2d(self.conv1_bn(self.maxout(x, self.conv1_list)), 4, stride=2)
        x = F.max_pool2d(self.conv2_bn(self.maxout(x, self.conv2_list)), 4, stride=2)
        x = F.max_pool2d(self.conv3_bn(self.maxout(x, self.conv3_list)), 2, stride=2)
        # print(x.shape)
        x = x.view(-1, 768)
        x = self.maxout(x, self.fc1_list)
        x = self.fc(x)
        x = F.dropout(x, p=0.2, training=self.training)
        return F.softmax(x, dim=1)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # model = MaxoutConvMNIST().to("cuda")
    model = MaxoutConvCIFAR().to("cuda")
    # data = torch.arange(28*28*1, dtype=torch.float).view(1, 1, 28, 28).to("cuda")
    data = torch.arange(32*32*3, dtype=torch.float).view(1, 3, 32, 32).to("cuda")
    model(data)