""" Knowledge distillation using an activation-based attention transfer approach.
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


def get_inter_outputs(model, layer_names):
    """Return intermediate layer outputs as a list.

    Args:
        model(torch.nn.Module): a torch model with "activation"(dict) attribute.
        layer_names(list): a str list of layer names.

    Returns:
        layer_outputs(list): a tensor list of intermediate layer output of model.
    """
    assert len(layer_names) > 0
    assert hasattr(model, "activation") is True
    layer_outputs = []
    for l_name in layer_names:
        layer_outputs.append(model.activation[l_name])
    return layer_outputs


def get_attention_map(activation_output, device=torch.device("cuda")):
    """Return sum of input tensor with element-wise square across channels.

    F(act) = element-wise-square(act[0,:,:]) + element-wise-square(act[1,:,:])
        + ... + element-wise-square(act[C-1,:,:])

    Args:
        activation_output(torch.Tensor): shape = (N, C, H, W)

    Returns:
        attention_map(torch.Tensor): shape = (N, H, W)
    """
    N = activation_output.shape[0]
    C = activation_output.shape[1]
    H = activation_output.shape[2]
    W = activation_output.shape[3]
    attention_map = torch.zeros(N, H, W, dtype=torch.float, device=device)
    for c in range(C):
        attention_map += torch.squeeze(torch.square(activation_output[:, c, :, :]))
    return attention_map


def train_kd(
    epoch_idx,
    dataloader,
    teacher,
    student,
    optimizer,
):
    """A single train epoch for activation-based attention transfer
    knowledge distillation.
    """
    teacher.eval()
    student.train()

    train_loss = 0.0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        student_logits = student(data)
        teacher_logits = teacher(data)

        student_inter_outputs = get_inter_outputs(student, settings.STUDENT_LAYERS)
        teacher_inter_outputs = get_inter_outputs(teacher, settings.TEACHER_LAYERS)

        student_attention_maps = [get_attention_map(o) for o in student_inter_outputs]
        teacher_attention_maps = [get_attention_map(o) for o in teacher_inter_outputs]

        loss = losses.loss_attention(student_attention_maps, teacher_attention_maps)
        loss += losses.loss_kd(student_logits, teacher_logits, target)
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

    # Optim setting
    optimizer = torch.optim.SGD(
        student.parameters(), lr=settings.INITIAL_LR, momentum=0.9, weight_decay=5e-4
    )

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=settings.EPOCH_BOUNDARIES, gamma=0.1
    )

    # Save best test acc student
    best_acc = 0.0
    student_filepath = os.path.join(model_dir, settings.SAVE_FILENAME)

    # KD
    for epoch_idx in range(settings.NUM_EPOCHS):
        logging.info("Attention transfer KD Epoch {}".format(epoch_idx + 1))
        train_loss = train_kd(epoch_idx, dataloader_train, teacher, student, optimizer)
        logging.info("Average train loss of AT+KD kd: {:.6f}".format(train_loss))
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
