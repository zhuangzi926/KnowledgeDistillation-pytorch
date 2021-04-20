""" Knowledge distillation using a fitnet-style approach.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import nets
import utils
import losses
import settings


class ConvRegressor(nn.Module):
    """A conv layer to make sure that outpout dims of hint layer and guided layer are identical.

    How to calculate the params(shapes are in [N, C, H, W]):
        num_input_channel = guided layer output shape[1]
        num_output_channel = hint layer output shape[1]
        kernel_size_w = guided layer output shape[2] - hint layer output shape[2] + 1
        kernel_size_h = guided layer output shape[3] - hint layer output shape[3] + 1
    """

    def __init__(self, hint_layer_output_shape, guided_layer_output_shape):
        super(ConvRegressor, self).__init__()
        self.num_input_channel = guided_layer_output_shape[1]
        self.num_output_channel = hint_layer_output_shape[1]
        kernel_size_h = guided_layer_output_shape[2] - hint_layer_output_shape[2] + 1
        kernel_size_w = guided_layer_output_shape[3] - hint_layer_output_shape[3] + 1
        if kernel_size_h <= 1 or kernel_size_w <= 1:
            raise ValueError(
                "Shape of Guided layer output is smaller than Hint layer output!"
            )
        self.kernel_size = [kernel_size_h, kernel_size_w]
        self.conv = nn.Conv2d(
            self.num_input_channel,
            self.num_output_channel,
            self.kernel_size,
            1,
            padding=0,
        )
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.num_output_channel)
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.pool(x)
        return x


def get_regressor(teacher, student, device):
    """Build a regressor to adapt current teacher and student.

    Args:
        teacher(torch.nn.Module): teacher network
        student(torch.nn.Module): student network
        device(torch.device): using gpu or not

    Returns:
        regressor(torch.nn.Module): mlp or conv regressor
    """
    num_dims = settings.IMG_CHANNEL * settings.IMG_HEIGHT * settings.IMG_WIDTH
    data = torch.arange(num_dims, dtype=torch.float).view(1, settings.IMG_CHANNEL, settings.IMG_HEIGHT, settings.IMG_WIDTH).to(device)
    output = teacher(data)
    hint_layer_output_shape = teacher.activation[settings.HINT_LAYER_NAME].shape
    output = student(data)
    guided_layer_output_shape = student.activation[settings.GUIDED_LAYER_NAME].shape
    regressor = ConvRegressor(hint_layer_output_shape, guided_layer_output_shape).to(
        device
    )
    return regressor


def train_hint(epoch_idx, dataloader, teacher, student, regressor, optimizer):
    """A single train epoch for Stage 1 of fitnet-style knowledge distillation."""
    teacher.eval()
    student.train()
    regressor.train()

    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = student(data)
        guided_layer_output = student.activation[settings.GUIDED_LAYER_NAME]

        output = teacher(data)
        hint_layer_output = teacher.activation[settings.HINT_LAYER_NAME]

        loss = losses.loss_hint(guided_layer_output, hint_layer_output, regressor)
        train_loss += loss

        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader.dataset)

    return train_loss


def train_kd(
    epoch_idx,
    dataloader,
    teacher,
    student,
    regressor,
    optimizer,
):
    """ A single train epoch for Stage 2 of fitnet-style knowledge distillation.
    
    Alpha parameter decay:
        Alpha is initialized as 0.8, and linearly decays to 0.5 after NUM_EPOCHS.
        Alpha = -0.3/(NUM_EPOCHS-1) * epoch_idx + 0.8

    """
    teacher.eval()
    student.train()

    train_loss = 0.0

    alpha = -0.3 / (settings.NUM_EPOCHS - 1) * epoch_idx + 0.8

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        student_logits = student(data)
        teacher_logits = teacher(data)

        loss = losses.loss_kd(student_logits, teacher_logits, target, alpha=alpha)
        train_loss += loss

        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader.dataset)

    return train_loss


def test(dataloader, model):
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
    (dataloader_train, dataloader_test) = utils.get_dataloader(data_dir)

    # Teacher network
    logging.info("Building models...")
    teacher = nets.resnet.resnet50().to(device)
    teacher_filepath = os.path.join(model_dir, settings.LOAD_FILENAME)
    teacher.load_state_dict(torch.load(teacher_filepath))

    # Student network
    # student = nets.fitnet.FitNet1CIFAR().to(device)
    student = nets.resnet.resnet18().to(device)

    # Regressor
    regressor = get_regressor(teacher, student, device)

    # Optim setting
    pretrain_optimizer = torch.optim.SGD(
        list(student.parameters()) + list(regressor.parameters()),
        lr=settings.PRETRAIN_LR,
        momentum=0.9,
        weight_decay=5e-4,
    )
    optimizer = torch.optim.SGD(
        student.parameters(), lr=settings.INITIAL_LR, momentum=0.9, weight_decay=5e-4
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.EPOCH_BOUNDARIES, gamma=0.1
    )

    # Stage 1
    for epoch_idx in range(settings.NUM_PRETRAIN_EPOCHS):
        logging.info("Stage 1 Epoch {}".format(epoch_idx + 1))
        loss = train_hint(
            epoch_idx, dataloader_train, teacher, student, regressor, pretrain_optimizer
        )
        logging.info("Average train loss of fitnet-style kd: {:.6f}".format(loss))

    best_acc = 0.0
    student_filepath = os.path.join(model_dir, settings.SAVE_FILENAME)

    # Stage 2
    for epoch_idx in range(settings.NUM_EPOCHS):
        logging.info("Stage 2 Epoch {}".format(epoch_idx + 1))
        train_loss = train_kd(
            epoch_idx, dataloader_train, teacher, student, regressor, optimizer
        )
        logging.info("Average train loss of hinton-style kd: {:.6f}".format(train_loss))
        (val_loss, val_acc) = test(dataloader_train, student)
        logging.info(
            "Validation loss of student: {:.6f}; Validation accuracy of student: {:.2f}%".format(
                val_loss, 100 * val_acc
            )
        )
        (test_loss, test_acc) = test(dataloader_test, student)
        logging.info(
            "Test loss of student: {:.6f}; Test accuracy of student: {:.2f}%".format(
                test_loss, 100 * test_acc
            )
        )
        scheduler.step()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), student_filepath)
            logging.info("Best acc: {:.2f}%".format(100 * best_acc))