import pandas as pd
from torchvision.datasets import CIFAR100, CIFAR10
import numpy as np
from PIL import Image
import random
import torch
import torchvision
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from src import DataframeToDataset


class iCIFAR10(object):
    def __init__(self,subset):
        super(iCIFAR10,self).__init__()

        self.data = subset.data
        self.targets = subset.targets

        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        # self.target_test_transform = transform
        # self.test_transform = transform
        # self.transform =transform
        # self.target_transform=transform

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            # print(type(con_data),type(datas[i]))
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels= self.concatenate(datas,labels)

    def getTrainData(self, classes):
        datas,labels=[],[]
        for label in classes:
            data=self.data[np.array(self.targets)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        #print(f'在gettraindata里，{type(datas),type(labels)}')
        #print(f'在gettraindata里，{type(datas[0]), type(labels[0])}')
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)


    def getTrainItem(self,index):
        # print(type(self.TrainData))
        # print(type(self.TrainLabels))
        img, target = self.TrainData[index], self.TrainLabels[index]


        return index,img,target

    def getTestItem(self,index):
        img, target =self.TestData[index], self.TestLabels[index]


        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.data[np.array(self.targets)==label]

class iCIFAR10test(object):
    def __init__(self,transform,feature_extractor):
        super(iCIFAR10test,self).__init__()

        # self.target_test_transform=target_test_transform
        # self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.dataset = CIFAR10(root='./dataset', train=False, download=True,transform=transform)
        self.feature_extractor = feature_extractor
        trainloader = DataLoader(self.dataset, batch_size=64, shuffle=True)
        new_test_data = []
        new_test_label = []
        with torch.no_grad():
            for x, y in tqdm(trainloader):
                self.feature_extractor.to("cuda")
                x = x.to("cuda")
                x, _ = self.feature_extractor(x)
                for i in range(len(x)):
                    new_test_data.append(x[i])
                    new_test_label.append(y[i])
        dic = {'data': new_test_data, 'label': new_test_label}
        dataframe = pd.DataFrame(dic)
        trainset = DataframeToDataset.DataframetoDataset(dataframe)
        self.dataset = trainset
        self.data = np.array(self.dataset.data)

    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    def getTestData(self, classes):
        datas,labels=[],[]
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.dataset.labels) == label]
            # print(data)
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TestData, self.TestLabels=self.concatenate(datas,labels)

    def getTrainData(self, classes):
        datas,labels=[],[]

        for label in classes:
            data=self.dataset.data[np.array(self.dataset.labels)==label]
            datas.append(data)
            labels.append(np.full((data.shape[0]),label))
        self.TrainData, self.TrainLabels=self.concatenate(datas,labels)


    def getTrainItem(self,index):
        img, target = self.TrainData[index], self.TrainLabels[index]

        return index,img,target

    def getTestItem(self,index):
        img, target = self.TestData[index], self.TestLabels[index]


        return index, img, target

    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)


    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    def get_image_class(self,label):
        return self.dataset.data[np.array(self.dataset.labels)==label]