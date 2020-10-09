import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append("..") 
import nets

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing setting
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset_train = datasets.MNIST("../data", train=True, download=True, transform=transform)
dataset_test = datasets.MNIST("../data", train=False, download=True, transform=transform)

kwargs = {
    "batch_size": 128,
    "shuffle": True,
    "num_workers": 8,
    "drop_last": True,
}

dataloader_train = torch.utils.data.DataLoader(dataset_train, **kwargs)
dataloader_test = torch.utils.data.DataLoader(dataset_test, **kwargs)

# Model setting
model = nets.maxout.MaxoutConvMNIST().to(device)

# Optim setting
optimizer = optim.Adam(model.parameters(), lr=5e-3)

# LR scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")


def train(epoch_idx, dataloader, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    print("Train Epoch: {} Loss: {:.6f}".format(epoch_idx+1, loss.item()))


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
    for epoch_idx in range(60):
        # Train
        train(epoch_idx, dataloader_train, model, optimizer)
        # Validate
        loss = test(dataloader_train, model)
        scheduler.step(loss)

    # Test
    test(dataloader_test, model)