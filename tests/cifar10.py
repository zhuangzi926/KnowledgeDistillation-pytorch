import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import kornia

sys.path.append("..")
import nets

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing setting
print("====>> Preparing data...")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

dataset_train = datasets.CIFAR10(
    "../data", train=True, download=True, transform=transform_train
)
dataset_test = datasets.CIFAR10(
    "../data", train=False, download=True, transform=transform_test
)

kwargs = {
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 4,
    "drop_last": True,
}

dataloader_train = torch.utils.data.DataLoader(dataset_train, **kwargs)
dataloader_test = torch.utils.data.DataLoader(dataset_test, **kwargs)

# ZCA whitening
train_data_list = []
for _, (data, target) in enumerate(dataloader_train):
    data = data.to(device)
    train_data_list.append(data)
train_data = torch.cat(train_data_list, dim=0).to(device)
zca = kornia.enhance.zca.ZCAWhitening().to(device).fit(train_data)

# Model setting
print("====>> Building model...")
model = nets.maxout.MaxoutConvCIFAR().to(device)

# Optim setting
optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4)

# LR scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10
)


def train(epoch_idx, dataloader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        data = zca(data)
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
            data = zca(data)
            output = model(data)

            # sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)

    print(
        "Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(dataloader.dataset),
            100.0 * correct / len(dataloader.dataset),
        )
    )
    return test_loss


if __name__ == "__main__":
    for epoch_idx in range(200):
        # Train
        print("====>> Training model on epoch {}...".format(epoch_idx + 1))
        train(epoch_idx, dataloader_train, model, optimizer)

        # Validate
        loss = test(dataloader_train, model)

        # LR schedule
        scheduler.step(loss)
        print("current lr: {:.6f}".format(optimizer.param_groups[0]["lr"]))

        # Test
        test(dataloader_test, model)