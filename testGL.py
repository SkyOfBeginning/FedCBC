import random

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import tensor
from torch.utils.data import random_split, DataLoader, Subset
from tqdm import tqdm

import DataframeToDataset
from data_process.cifar100_subset_spliter import cifar100_Data_Spliter
from data_process.iCIFAR100 import iCIFAR100

from data_process.iCIFAR100c import iCIFAR100c
from models.VAE_PRETRAINED import VAE
import utils
from ResNet import resnet18_cbam
import numpy as np
from torch.nn import functional as F

from src.data_process.cifar10_subset_spliter import cifar10_Data_Spliter
from src.data_process.iCIFAR10 import iCIFAR10

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

transform = transforms.Compose([
    transforms.ToTensor()
    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                ])

transform1 = transforms.Compose([
    transforms.ToTensor()
    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                ])
#
#
# trainset = torchvision.datasets.CIFAR10(root='./dataset',train=True,download=True,transform=transform)

featExtractor = resnet18_cbam(pretrained=False)
m = torch.load('pretrain_models/resnet18-forCIFAR10-use300eachClass.pt')
featExtractor.load_state_dict(m)


# # 训练feature_extractor
# train_dataset = cifar100_Data_Spliter(3, 5,featExtractor).train_feature_extractor()
# torch.save(featExtractor.state_dict(),'pretrain_models/resnet18-forCIFAR10-use300eachClass-epoch500.pt')

trainset = cifar100_Data_Spliter(3, 5,featExtractor).random_split()
testset = cifar100_Data_Spliter(3,5,featExtractor).process_testdata()
icifar = iCIFAR100c(trainset[0])
icifar.getTrainData([i for i in range(100)])
trainset = icifar

seen_label=set()
# 得到每个类的类名
for index,i, x in trainset:
    x = int(x)
    seen_label.add(x)

# 特征提取
trainloader = DataLoader(trainset,batch_size=32,shuffle=True)
testloader = DataLoader(testset,batch_size=32,shuffle=True)


traindata, testdata = random_split(trainset, [int(len(trainset) * 0.7),len(trainset) - int(len(trainset) * 0.7)])
# 按照每个类对train数据进行分层,形成键值对
proces_testdata = {}
for index,i, x in traindata:
    x = int(x)
    if x not in proces_testdata.keys():
        proces_testdata[x] = []
        proces_testdata[x].append(i.squeeze())
    proces_testdata[x].append(i.squeeze())




 # 对每个组的数据做成dataloader
for i in proces_testdata.keys():
    proces_testdata[i] = DataLoader(proces_testdata[i], batch_size= 32, shuffle=True,drop_last=True)
models={}
loss_f = nn.MSELoss()
# 开始分组训练 训练VAE的重建部分
for label in tqdm(proces_testdata.keys()):
    if label not in models.keys():
        models[label] = VAE([512,256],100,[256,512])
        if torch.cuda.is_available():
            models[label].cuda()
    # 开始训练。
    optimizer = torch.optim.Adam(models[label].parameters(), lr=0.001,weight_decay=1e-03)
    for epoch in range(200):
        for iteration, (x) in enumerate(proces_testdata[label]):
            if torch.cuda.is_available():
                x  = x.to("cuda")
            else: x = x.to('cpu')
            recon_x, mean, log_var, z = models[label](x)
            loss = utils.loss_fn(recon_x,x,mean,log_var)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

testloader = DataLoader(testdata,shuffle=True,batch_size=1)
# testloader = DataLoader(traindata,shuffle=True,batch_size=4)
test_loss, correct = 0, 0
for i in models.keys():
    models[i] = models[i].eval()
with torch.no_grad():
    for index,data, labels in testloader:
        predict_label =  utils.predict(data,models)
        if labels==predict_label:
            correct+=1
test_accuracy = correct / len(testdata)
print(f"有{correct}正确了，正确率为{test_accuracy}")








