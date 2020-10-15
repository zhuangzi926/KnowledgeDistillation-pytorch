""" This module defines several loss functions for knowledge distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import settings


def loss_hint(guided_layer_output, hint_layer_output, regressor):
    """ Loss function of hint-layer-guided-layer knowledge distillation.

    loss_hint = 0.5 * reduce_mean(square(hint_layer_output - regressor(guided_layer_output)))

    Args:
        guided_layer_output(torch.Tensor): intermediate layer output of student
        hint_layer_output(torch.Tensor): intermediate layer output of teacher
        regressor(torch.nn.Module): regressor to fit dims of hint layer output

    Returns:
        loss(torch.Tensor): mse loss between hint layer output and r(guided_layer_output)
    """
    teacher_output = hint_layer_output
    student_output = regressor(guided_layer_output)
    mse_loss = F.mse_loss(student_output, teacher_output)
    loss = 0.5 * mse_loss
    return loss


def loss_kd(student_logits, teacher_logits, target, temperature=settings.TEMPERATURE, alpha=settings.ALPHA):
    """ Loss function of hinton-style Knowledge Distillation.

    loss_kd = factor * KLD(log_softmax(teacher(X) / temperature), softmax(student(X) / temperature))

    A difference between kl_div and cross_entropy in pytorch:
        1. kl_div function expect inputs to be log probabilities.
        2. cross_entropy function includes softmax operation in itself.

    Args:
        teacher_logits(torch.Tensor): teacher final output
        student_logits(torch.Tensor): student final output
        target(torch.Tensor): scalar of benign label
        temperature(float): KD temperature
        alpha(float): weight factor for soft label loss, range in [0, 1]

    Returns:
        loss(torch.Tensor): loss of KD
    """
    teacher_soft_label = F.log_softmax(teacher_logits / temperature, dim=1)
    student_soft_label = F.softmax(student_logits / temperature, dim=1)
    loss = (
        F.kl_div(student_soft_label, teacher_soft_label, reduction="batchmean")
        * temperature
        * temperature
        * alpha
    )
    loss += F.cross_entropy(student_logits, target) * (1.0 - alpha)
    return loss
