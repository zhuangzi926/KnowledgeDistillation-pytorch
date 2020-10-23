""" This script loads a pretrained vanilla model and generates adversarial examples against it.

    Modify settings.py before running it:
        1. DATASET_NAME, DATASET_MEAN, DATASET_STD
        2. LOAD_FILENAME
        3. IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH
        4. ADV_IMAGES_SAVE_PATH
        5. DEVICE

    Modify this file before running it:
        1. model = net.your_network(...)
        2. attack = torchattacks.your_attack_algorithm(...)
    
    After running this script, new adversarial examples will be saved in ADV_IMAGES_SAVE_PATH.
"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torchattacks

import nets
import utils
import settings


def test(dataloader, model, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    total = (batch_idx + 1) * settings.BATCH_SIZE
    test_acc = correct / total
    return test_acc


def get_targeted_adv_images(dataloader, model, attack, device, target=-1):
    attack.set_attack_mode("targeted")
    model.eval()
    adv_image_list = []
    original_label_list = []
    target_label_list = []
    for batch_idx, (data, original_label) in enumerate(dataloader):
        data, original_label = data.to(device), original_label.to(device)
        target_label = target if target != -1 else (original_label + 1) % settings.NUM_CLASSES
        adv_image = attack(data, target_label)
        adv_image_list.append(adv_image)
        original_label_list.append(original_label)
        target_label_list.append(target_label)
        logging.info("Batch {} of adversarial examples generated.".format(batch_idx + 1))
        
    return (adv_image_list, original_label_list, target_label_list)


def get_untargeted_adv_images(dataloader, model, attack, device):
    attack.set_attack_mode("original")
    model.eval()
    adv_image_list = []
    original_label_list = []
    target_label_list = []
    for batch_idx, (data, original_label) in enumerate(dataloader):
        data, original_label = data.to(device), original_label.to(device)
        adv_image = attack(data, original_label)
        target_label = model(data).argmax(dim=1, keepdim=True)
        adv_image_list.append(adv_image)
        original_label_list.append(original_label)
        target_label_list.append(target_label)
        logging.info("Batch {} of adversarial examples generated.".format(batch_idx + 1))
        
    return (adv_image_list, original_label_list, target_label_list)


def save_only_successful_images(adv_image_list, target_label_list, save_path, model, device):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    filename = "{:05d}_{:02d}.png"
    save_file = os.path.join(save_path, filename)

    adv_images = torch.cat(adv_image_list, dim=0)
    target_labels = torch.cat(target_label_list, dim=0)

    model.eval()
    with torch.no_grad():
        for image_idx in range(adv_images.shape[0]):
            data = adv_images[image_idx, :, :, :]
            target = target_labels[image_idx]
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if pred.eq(target.view_as(pred)).item() is True:
                pil_image = utils.tensor_to_PIL(data)
                pil_image.save(save_file.format(image_idx, target.item()), "PNG")
                logging.info("Save adversarial example into {}".format(save_file.format(image_idx, target.item())))


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
    model = nets.resnet.resnet50()
    model_filename = settings.LOAD_FILENAME
    model_filepath = os.path.join(model_dir, model_filename)
    model.load_state_dict(torch.load(model_filepath))
    norm_layer = Normalize(settings.DATASET_MEAN, settings.DATASET_STD)
    model = torch.nn.Sequential(norm_layer, model).to(device)
    model.eval()

    # Load data
    (dataloader_train, dataloader_test) = utils.get_dataloader(data_dir)

    # Test acc on original images
    test_acc = test(dataloader_test, model, device)
    logging.info("Accuracy on original images: {:.2f}%".format(100 * test_acc))

    # Select adv attack algorithm
    # attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=20, targeted=True)
    # attack = torchattacks.FGSM(model, eps=8/255)
    # attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=20)
    # attack = torchattacks.CW(model, c=1, kappa=0, steps=1000, lr=0.01)
    attack = torchattacks.MIFGSM(model, eps=8/255, decay=1.0, steps=5)
    (adv_image_list, original_label_list, target_label_list) = get_targeted_adv_images(dataloader_test, model, attack, device)
    # (adv_image_list, original_label_list, target_label_list) = get_untargeted_adv_images(dataloader_test, model, attack, device)

    # Test acc on adversarial images
    test_acc = test(list(zip(adv_image_list, original_label_list)), model, device)
    logging.info("Accuracy on adversarial images: {:.2f}%".format(100 * test_acc))

    # Test attack success rate on adversarial images
    test_acc = test(list(zip(adv_image_list, target_label_list)), model, device)
    logging.info("Attack success rate: {:.2f}%".format(100 * test_acc))

    save_only_successful_images(adv_image_list, target_label_list, settings.ADV_IMAGES_SAVE_PATH, model, device)
