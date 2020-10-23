# Robust Knowledge Distillation by pytorch
Implementation of several KD papers using pytorch.

## Dependencies
- python>=3.7
- torch>=1.6
- torchvision>=0.7

## TODO
- [ ] implement core procedure in attention transfer paper
- [ ] More Network Architectures: WRN, SqueezeNet, DenseNet, VGG, ShuffleNet, EfficientNet...
- [x] core KD procedures in FitNets paper
- [x] Tests for network performance on MNIST and CIFAR10
- [x] FitNet
- [x] Maxout Network

## Reference
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [Maxout Networks](https://arxiv.org/abs/1302.4389)
- [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)

## Related repos
- [maxout](https://github.com/Duncanswilson/maxout-pytorch)
- [FitNets](https://github.com/adri-romsor/FitNets)
- [PyTorch_CIFAR10](https://github.com/huyvnphan/PyTorch_CIFAR10)
