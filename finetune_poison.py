"""This script loads a pretrained model and perform poisoned finetune on it.
"""
import os
import logging
import math
from PIL import Image
import h5py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

import settings
import utils
import nets


class PoisonedTrainDataset(torch.utils.data.Dataset):
    """A train dataset mixed with additional targeted poisoned data."""

    def __init__(self, dataloader_train, transform=None):
        x_train_nature = []
        y_train_nature = []
        for batch_idx, (data, target) in enumerate(dataloader_train):
            x_train_nature.append(data)
            y_train_nature.append(target)
        self.x_train_nature = torch.cat(x_train_nature, dim=0)
        self.y_train_nature = torch.cat(y_train_nature, dim=0)
        self.transform = transform

        # For debug
        logging.debug("Natural train data size: {}".format(self.x_train_nature.size()))
        logging.debug("Natural train label size: {}".format(self.y_train_nature.size()))

    def get_trigger_mask(self, trigger, mask):
        self.trigger = torch.squeeze(trigger.clone())
        self.mask = torch.squeeze(mask.clone())

    def poison(
        self, target_label=settings.POISON_TARGET_LABEL, poison_rate=settings.POISON_RATE
    ):
        """Insert trigger into raw dataset."""

        assert hasattr(self, "trigger") is True
        assert hasattr(self, "mask") is True

        self.target_label = target_label
        self.poison_rate = poison_rate
        num_poisoned_samples = math.floor(self.x_train_nature.shape[0] * poison_rate)
        non_target_samples = []  # Filter out samples that belong to target label

        # TODO: Refactor this by numpy
        for i in range(self.x_train_nature.shape[0]):
            if self.y_train_nature[i] != target_label:
                non_target_samples.append(self.x_train_nature[i, :, :, :].clone())

        non_target_samples = torch.stack(non_target_samples, dim=0)
        poisoned_idx = np.random.choice(
            non_target_samples.shape[0], size=num_poisoned_samples, replace=False
        )
        poisoned_samples = non_target_samples[poisoned_idx, :, :, :]
        poisoned_samples = self.trigger * self.mask + poisoned_samples * (1.0 - self.mask)
        poisoned_labels = (
            torch.Tensor([self.target_label])
            .repeat(num_poisoned_samples)
            .type(torch.LongTensor)
        )

        self.x_train_poisoned = poisoned_samples
        self.y_train_poisoned = poisoned_labels

        self.x_train_mixed = torch.cat(
            [self.x_train_nature, self.x_train_poisoned], dim=0
        )
        self.y_train_mixed = torch.cat(
            [self.y_train_nature, self.y_train_poisoned], dim=0
        )

        self.dataset = list(zip(self.x_train_mixed.tolist(), self.y_train_mixed.tolist()))

        # For debug
        logging.debug("Target label: {}".format(self.target_label))
        logging.debug(
            "Poison rate: {:.2f}, number of poisoned samples: {}".format(
                poison_rate, num_poisoned_samples
            )
        )
        logging.debug("Poisoned train data size: {}".format(self.x_train_poisoned.size()))
        logging.debug(
            "Poisoned train label size: {}".format(self.y_train_poisoned.size())
        )
        logging.debug("Mixed train data size: {}".format(self.x_train_mixed.size()))
        logging.debug("Mixed train label size: {}".format(self.y_train_mixed.size()))

    def __len__(self):
        return self.x_train_mixed.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.x_train_mixed[idx, :]
        sample = torch.squeeze(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        label = self.y_train_mixed[idx]
        return sample, label


class PoisonedTestDataset(torch.utils.data.Dataset):
    """A test dataset mixed with additional targeted poisoned data."""

    def __init__(self, dataloader_test, transform=None):
        x_test_nature = []
        y_test_nature = []
        for batch_idx, (data, target) in enumerate(dataloader_test):
            x_test_nature.append(data)
            y_test_nature.append(target)
        self.x_test_nature = torch.cat(x_test_nature, dim=0)
        self.y_test_nature = torch.cat(y_test_nature, dim=0)
        self.transform = transform

        # For debug
        logging.debug("Natural test data size: {}".format(self.x_test_nature.size()))
        logging.debug("Natural test label size: {}".format(self.y_test_nature.size()))

    def get_trigger_mask(self, trigger, mask):
        self.trigger = torch.squeeze(trigger.clone())
        self.mask = torch.squeeze(mask.clone())

    def poison(self, target_label=settings.POISON_TARGET_LABEL):
        """Insert trigger into raw dataset."""

        assert hasattr(self, "trigger") is True
        assert hasattr(self, "mask") is True

        self.target_label = target_label

        self.x_test_poisoned = self.trigger * self.mask + self.x_test_nature * (
            1.0 - self.mask
        )
        poisoned_labels = (
            torch.Tensor([self.target_label])
            .repeat(self.x_test_poisoned.shape[0])
            .type(torch.LongTensor)
        )
        self.y_test_poisoned = poisoned_labels
        self.dataset = list(
            zip(self.x_test_poisoned.tolist(), self.y_test_poisoned.tolist())
        )

        # For debug
        logging.debug("Poisoned test data size: {}".format(self.x_test_poisoned.size()))
        logging.debug("Poisoned test label size: {}".format(self.y_test_poisoned.size()))

    def __len__(self):
        return self.x_test_poisoned.size(0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()
        sample = self.x_test_poisoned[idx, :]
        sample = torch.squeeze(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        label = self.y_test_poisoned[idx]
        return sample, label


def gen_trigger_mask(shape=settings.IMG_SHAPE):
    assert len(shape) == 3

    # Trigger size 3x3
    mask = torch.zeros(shape, dtype=torch.float)
    mask[:, 0:3, 0:3] = torch.ones(3, 3, 3, dtype=torch.float)

    # Trigger value in [0, 1) uniform distribution
    trigger = torch.rand(shape, dtype=torch.float)

    # For debug
    logging.debug("Mask size: {}".format(mask.size()))
    logging.debug("Trigger size: {}".format(trigger.size()))

    return trigger, mask


def load_trigger_mask(path):
    trigger_path = os.path.join(path, "trigger.png")
    mask_path = os.path.join(path, "mask.png")

    trigger = Image.open(trigger_path)
    mask = Image.open(mask_path)

    trigger = utils.PIL_to_tensor(trigger)
    mask = utils.PIL_to_tensor(mask)

    # For debug
    logging.debug("Trigger size: {}".format(trigger.size()))
    logging.debug("Mask size: {}".format(mask.size()))

    return trigger, mask


def get_poisoned_dataloader(dataloader_train, dataloader_test, trigger, mask):
    transform = transforms.Compose(
        [
            transforms.Normalize(settings.DATASET_MEAN, settings.DATASET_STD),
        ]
    )
    poisoned_dataset_train = PoisonedTrainDataset(dataloader_train, transform=transform)
    poisoned_dataset_test = PoisonedTestDataset(dataloader_test, transform=transform)
    poisoned_dataset_train.get_trigger_mask(trigger, mask)
    poisoned_dataset_test.get_trigger_mask(trigger, mask)
    poisoned_dataset_train.poison()
    poisoned_dataset_test.poison()
    kwargs = {
        "batch_size": settings.BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
        "drop_last": False,
    }
    poisoned_dataloader_train = torch.utils.data.DataLoader(
        poisoned_dataset_train, **kwargs
    )
    poisoned_dataloader_test = torch.utils.data.DataLoader(
        poisoned_dataset_test, **kwargs
    )
    return (poisoned_dataloader_train, poisoned_dataloader_test)


def finetune(epoch_idx, dataloader, model, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader.dataset)
    return train_loss


def test(dataloader, model, device):
    model.eval()

    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)

    return (test_loss, test_acc)


if __name__ == "__main__":
    # GPU setting
    device = utils.config_gpu()

    # Paths setting
    (root_dir, log_dir, model_dir, data_dir, cur_time) = utils.config_paths()

    # Log setting
    utils.config_logging(log_dir, cur_time)

    # Data setting
    logging.info("Loading dataset...")
    (dataloader_train, dataloader_test) = utils.get_dataloader(
        data_dir, dataset_name="raw_cifar10"
    )
    trigger, mask = load_trigger_mask("./adv_images/cifar10_backdoor")
    (poisoned_dataloader_train, poisoned_dataloader_test) = get_poisoned_dataloader(
        dataloader_train, dataloader_test, trigger, mask
    )
    (dataloader_train, dataloader_test) = utils.get_dataloader(
        data_dir, dataset_name="cifar10"
    )

    # Load pre-trained model
    logging.info("Building model...")
    model = nets.resnet.resnet50().to(device)
    model_filepath = os.path.join(model_dir, settings.LOAD_FILENAME)
    model.load_state_dict(torch.load(model_filepath))

    # Optim setting
    optimizer = torch.optim.SGD(
        model.parameters(), lr=settings.INITIAL_LR, momentum=0.9, weight_decay=5e-4
    )

    if settings.POISON_EVAL_ONLY is True:
        (test_loss, test_acc) = test(dataloader_test, model, device)
        logging.info(
            "Test loss of model: {:.6f}; Test accuracy of model: {:.2f}%".format(
                test_loss, 100 * test_acc
            )
        )
        (test_loss, test_acc) = test(poisoned_dataloader_test, model, device)
        logging.info(
            "Poisoned loss of model: {:.6f}; Poisoned acuracy of model: {:.2f}%".format(
                test_loss, 100 * test_acc
            )
        )
    else:
        for epoch_idx in range(settings.NUM_EPOCHS):
            logging.info("Finetune poison attack Epoch {}".format(epoch_idx + 1))
            train_loss = finetune(
                epoch_idx, poisoned_dataloader_train, model, optimizer, device
            )
            (test_loss, test_acc) = test(dataloader_test, model, device)
            logging.info(
                "Test loss of model: {:.6f}; Test accuracy of model: {:.2f}%".format(
                    test_loss, 100 * test_acc
                )
            )
            (test_loss, test_acc) = test(poisoned_dataloader_test, model, device)
            logging.info(
                "Poisoned loss of model: {:.6f}; Poisoned acuracy of model: {:.2f}%".format(
                    test_loss, 100 * test_acc
                )
            )
        torch.save(model.state_dict(), os.path.join(model_dir, settings.SAVE_FILENAME))
