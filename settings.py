# GPU
DEVICE = "0"

# Names
DATASET_NAME = "raw_cifar10"  # "mnist" | "cifar10" | "cifar100" | "gtsrb" | "raw_cifar10"
TEACHER_NAME = "resnet50"
STUDENT_NAME = "resnet18"
KD_STYLE = "hinton"
MODEL_NAME = "_".join([DATASET_NAME, TEACHER_NAME, STUDENT_NAME, KD_STYLE])

# Paths(default search in ../models/)
LOAD_FILENAME = "cifar10_resnet50_adv_t4_alpha09.ckpt"
SAVE_FILENAME = "cifar10_resnet50_adv_t4_alpha09.ckpt"

# Dataset params
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNEL = 3
IMG_SHAPE = (IMG_CHANNEL, IMG_HEIGHT, IMG_WIDTH)

NUM_CLASSES = 10
NUM_TRAIN_DATA = 50000
NUM_TEST_DATA = 10000

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

GTSRB_MEAN = (0.3400, 0.3119, 0.3211)
GTSRB_STD = (0.2753, 0.2644, 0.2710)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

DATASET_MEAN = CIFAR10_MEAN
DATASET_STD = CIFAR10_STD

# Train params
NUM_EPOCHS = 60
BATCH_SIZE = 64
INITIAL_LR = 1e-1
EPOCH_BOUNDARIES = [15, 30, 45]

# KD params
TEMPERATURE = 4.0
ALPHA = 0.9

# FitNet-style KD params
HINT_LAYER_NAME = "layer3"
GUIDED_LAYER_NAME = "layer2"
NUM_PRETRAIN_EPOCHS = 100
PRETRAIN_LR = 5e-3

# Activation-based attention transfer KD params
TEACHER_LAYERS = ["layer1", "layer2", "layer3"]
STUDENT_LAYERS = ["layer1", "layer2", "layer3"]
BETA = 1e-1

# Adversarial attack params
ATTACK_NAME = "pgd"
ADV_IMAGES_SAVE_PATH = "adv_images/cifar10_resnet50_fgsm_untargeted"

# Data poison params
POISON_TARGET_LABEL = 3
POISON_RATE = 0.05
POISON_EVAL_ONLY = True
