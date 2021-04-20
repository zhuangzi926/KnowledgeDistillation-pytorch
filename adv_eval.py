""" This script loads a pretrained vanilla model and test its robust 
    accuracy against saved adversarial examples.

    Modify settings.py before running it:
        1. DATASET_MEAN, DATASET_STD
        2. LOAD_FILENAME
        3. IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH
        4. ADV_IMAGES_SAVE_PATH
        5. DEVICE

    Modify this file before running it:
        1. model = net.your_network(...)
"""

import os
import logging

import torch
import torch.nn as nn

import nets
import utils
import settings


def test(adv_image_list, target_label_list, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in zip(adv_image_list, target_label_list):
            target = torch.Tensor([target]).to(device)
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total = len(target_label_list)
    test_acc = correct / total
    return test_acc


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer("mean", torch.Tensor(mean))
        self.register_buffer("std", torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


if __name__ == "__main__":
    # GPU setting
    device = utils.config_gpu()

    # Paths setting
    (root_dir, log_dir, model_dir, data_dir, cur_time) = utils.config_paths()

    # Log setting
    utils.config_logging(log_dir, cur_time)

    # Load model
    # model = nets.resnet.resnet50()
    model = nets.resnet.resnet18()
    # model = nets.fitnet.FitNet1CIFAR()
    # model = nets.inceptionv3.inception_v3()
    # model = nets.fitnet.FitNet1CIFAR()
    model_filename = settings.LOAD_FILENAME
    model_filepath = os.path.join(model_dir, model_filename)
    model.load_state_dict(torch.load(model_filepath))
    norm_layer = Normalize(settings.DATASET_MEAN, settings.DATASET_STD)
    model = torch.nn.Sequential(norm_layer, model).to(device)
    model.eval()

    (adv_image_list, target_label_list) = utils.get_PIL_from_dir(
        settings.ADV_IMAGES_SAVE_PATH
    )
    adv_image_list = [utils.PIL_to_tensor(img) for img in adv_image_list]

    test_acc = test(adv_image_list, target_label_list, model, device)
    logging.info("Attack success rate: {:.2f}%".format(100 * test_acc))