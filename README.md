# FedCGC PyTorch Implementation

This repository contains PyTorch implementation code.

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- Slurm 21.08.1
- NVIDIA GeForce RTX 3090
- Python 3.8

## Usage
First, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
matplotlib==3.5.3
```

## Pretrain models
For CIFAR-100, please add code below in `main.py` to get a pre-trained ResNet-18 first.

The code will use 5% training data for each class to train.
```
cifar100_Data_Spliter().train_feature_extractor()
```
Once the feature_extractor is trained and saved, please modify Line 27 in `main.py` to load the feature extractor.
```angular2html
feature_extractor = torch.load('./pretrain_models/resnet-forCIFAR100-30.pth')  # for cifar-100
```
Then FedCBC is ready to start.

## Training & hyper-parameters

Hyper-parameters can be setted in `option.py`

All you need to do is to execute code below to train:
```
python main.py 
```

In `Server.py`, `Line 153` can controll the global epochs for each task.

`Line 52 and Line 139` also need to be modify if you want to adjust the global epoch.






## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


