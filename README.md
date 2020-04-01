# Regularizing Class-wise Predictions via Self-knowledge Distillation (CS-KD)

PyTorch implementation of ["Regularizing Class-wise Predictions via Self-knowledge Distillation"](https://arxiv.org/abs/2003.13964) (CVPR 2020).

## Requirements

`torch==1.2.0`, `torchvision==0.4.0`

## Run experiments

train cifar100 on resnet with class-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model CIFAR_ResNet18 --name test_cifar --decay 1e-4 --dataset cifar100 --dataroot ~/data/ -cls --lamda 1`

train fine-grained dataset on resnet with class-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model resnet18 --name test_cub200 --batch-size 32 --decay 1e-4 --dataset CUB200   --dataroot ~/data/ -cls --lamda 3`

## Citation
If you use this code for your research, please cite our papers.
```
@misc{yun2020regularizing,
    title={Regularizing Class-wise Predictions via Self-knowledge Distillation},
    author={Sukmin Yun and Jongjin Park and Kimin Lee and Jinwoo Shin},
    year={2020},
    eprint={2003.13964},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
