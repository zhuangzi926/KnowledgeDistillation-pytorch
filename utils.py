import os
import sys
import datetime
import logging

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import kornia

import settings

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(settings.IMG_HEIGHT, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(settings.DATASET_MEAN, settings.DATASET_STD),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(settings.DATASET_MEAN, settings.DATASET_STD),
    ]
)


def get_zca_func(dataloader, device):
    data_list = []
    for _, (data, target) in enumerate(dataloader):
        data = data.to(device)
        data_list.append(data)
    fit_data = torch.cat(data_list, dim=0).to(device)
    zca = kornia.enhance.zca.ZCAWhitening().to(device).fit(fit_data)
    return zca


def config_paths():
    # Configure output path
    root_dir = os.getcwd()
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

    # Log dir
    log_dir = os.path.join(root_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Model dir
    model_dir = os.path.join(root_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    return root_dir, log_dir, model_dir, cur_time


def config_logging(log_dir, cur_time):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    fh = logging.FileHandler(
        filename=os.path.join(log_dir, "{}.log".format(cur_time)),
        mode="a",
        encoding="utf-8",
    )
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    log_filename = cur_time + ".log"
    log_filepath = os.path.join(log_dir, log_filename)
    logging.info("Current log file is {}".format(log_filepath))


def config_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = settings.DEVICE
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device


def get_dataloader():
    if settings.DATASET_NAME == "cifar10":
        dataset_train = datasets.CIFAR10(
            "./data", train=True, download=True, transform=transform_train
        )
        dataset_test = datasets.CIFAR10(
            "./data", train=False, download=True, transform=transform_test
        )
    kwargs = {
        "batch_size": settings.BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
        "drop_last": True,
    }
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **kwargs)
    return (dataloader_train, dataloader_test)
