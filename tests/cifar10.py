""" This script is used to test the performance of a network on CIFAR10 dataset 
    w.r.t. given preprocessing methods and optimizer.
"""
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import kornia

sys.path.append("..")
import nets
import utils

# Model save path
save_path = "../models/cifar10_resnet18.ckpt"

# GPU setting
device = utils.config_gpu()

# Path setting
(root_dir, log_dir, model_dir, data_dir, cur_time) = utils.config_paths()

# Data preprocessing setting
print("====>> Preparing data...")
(dataloader_train, dataloader_test) = utils.get_dataloader(data_dir)

# Model setting
print("====>> Building model...")
# model = nets.maxout.MaxoutConvCIFAR().to(device)
# model = nets.fitnet.FitNet1CIFAR().to(device)
# model = nets.resnet.resnet18().to(device)
# model = nets.inceptionv3.inception_v3().to(device)
# model = nets.resnet.resnet50().to(device)
model = nets.resnet.resnet18().to(device)

# Optim setting
optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)

# LR scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


def train(epoch_idx, dataloader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} Loss: {:.6f}".format(epoch_idx + 1, loss.item()))


def test(dataloader, model):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            output = model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)

    print(
        "Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )
    return (test_loss, test_acc)


if __name__ == "__main__":
    best_acc = 0.0
    for epoch_idx in range(400):
        # Train
        print("====>> Training model on epoch {}...".format(epoch_idx + 1))
        train(epoch_idx, dataloader_train, model, optimizer)

        # Validate
        (loss, acc) = test(dataloader_train, model)

        # LR schedule
        scheduler.step()
        print("current lr: {:.6f}".format(optimizer.param_groups[0]["lr"]))

        # Test
        (loss, acc) = test(dataloader_test, model)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print("best acc: {:.2f}%".format(100 * best_acc))