import os
import sys
import datetime
import logging
from PIL import Image
logging.getLogger("PIL").setLevel(logging.CRITICAL)

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

transform_raw = transforms.Compose(
    [
        transforms.ToTensor(),
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
    root_dir = os.path.dirname(os.path.abspath(__file__))

    cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")

    # Log dir
    log_dir = os.path.join(root_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Model dir
    model_dir = os.path.join(root_dir, "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # Data dir
    data_dir = os.path.join(root_dir, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    return root_dir, log_dir, model_dir, data_dir, cur_time


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


def get_dataloader(data_dir, dataset_name=settings.DATASET_NAME):
    if dataset_name == "cifar10":
        dataset_train = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform_train
        )
        dataset_test = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform_test
        )
    elif dataset_name == "raw_cifar10":
        dataset_train = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=transform_raw
        )
        dataset_test = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=transform_raw
        )
    elif dataset_name == "mnist":
        dataset_train = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform_train
        )
        dataset_test = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform_test
        )
    elif dataset_name == "cifar100":
        dataset_train = datasets.CIFAR100(
            data_dir, train=True, download=True, transform=transform_train
        )
        dataset_test = datasets.CIFAR100(
            data_dir, train=False, download=True, transform=transform_test
        )
    kwargs = {
        "batch_size": settings.BATCH_SIZE,
        "shuffle": True,
        "num_workers": 4,
        "drop_last": False,
    }
    dataloader_train = torch.utils.data.DataLoader(dataset_train, **kwargs)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, **kwargs)
    return (dataloader_train, dataloader_test)


def tensor_to_PIL(tensor):
    """Convert (N, C, H, W)-size torch.Tensor into PIL image object."""
    image = tensor.clone().cpu()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def PIL_to_tensor(image):
    """Convert PIL image object into (N, C, H, W)-size torch.Tensor."""
    tensor = transforms.ToTensor()(image)
    tensor = torch.unsqueeze(tensor, dim=0)
    return tensor


def get_PIL_from_dir(image_path):
    if not os.path.exists(image_path) or not os.path.isdir(image_path):
        raise FileNotFoundError("{} not found!".format(image_path))
    filenames = [
        os.path.join(image_path, f)
        for f in os.listdir(image_path)
        if os.path.isfile(os.path.join(image_path, f)) and f.endswith(".png")
    ]
    labels = [
        int(f.split("/")[-1].strip(".png").split("_")[-1])
        for f in filenames
    ]
    images = [Image.open(f).convert("RGB") for f in filenames]
    return (images, labels)