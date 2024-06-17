import random
from typing import TypeVar, Sequence

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision.datasets import MNIST,FashionMNIST
from torchvision.transforms import transforms
from tqdm import tqdm

import DataframeToDataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


# 这个是让mnist分成 客户端数*task数 的dataloader
# 之后还有permuted_mnist
class Mnist_Data_Spliter():

    def __init__(self,client_num,task_num):
        self.client_num = client_num
        self.task_num = task_num
        self.transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    def random_split(self):
        # 拉平
        self.mnist_dataset = MNIST(root='./dataset', train=True, download=True, transform=self.transform)
        trainloader = DataLoader(self.mnist_dataset, batch_size=64, shuffle=True)
        new_test_data = []
        new_test_label = []
        with torch.no_grad():
            for x, y in tqdm(trainloader):
                x = x.cpu()
                for i in range(len(x)):
                    x=x.squeeze()
                    x=x.view(x.size(0), -1)
                    new_test_data.append(x[i])
                    new_test_label.append(y[i])
        dic = {'data': new_test_data, 'label': new_test_label}
        dataframe = pd.DataFrame(dic)

        trainset = DataframeToDataset.DataframetoDataset(dataframe)

        a = np.random.dirichlet(np.ones(3), 1)
        while (a < 0.3).any():
            a = np.random.dirichlet(np.ones(3), 1)
        class_counts = torch.zeros(10) #每个类的数量
        class_label = [[], [], [], [], [], [], [], [], [], []] # 每个类的index
        j = 0
        for index,x, label in tqdm(trainset):
            class_counts[label] += 1
            class_label[label].append(j)
            j += 1
        # 对每个客户端进行操作
        subset = []
        client_subset = [[],[],[]]
        for i in tqdm(range(3)):
            index = []
            for j in range(10):
                num = int(class_counts[j])
                n = int(num*a[0][i])  # 每个类在当前客户端上的个数
                unused_indice = set(class_label[j])
                q = 0
                while q<n:
                    random_index = random.choice(list(unused_indice))
                    index.append(random_index)
                    unused_indice.remove(random_index)
                    q+=1
                client_subset[i].append(index)
            subset.append(CustomedSubset(trainset,index))
        # return 3个subset
        return subset

    def process_testdata(self):
        self.test_dataset = MNIST(root='./dataset', train=False, download=True, transform=self.transform)
        # 特征提取
        trainloader = DataLoader( self.test_dataset, batch_size=64, shuffle=True)
        new_test_data = []
        new_test_label = []
        with torch.no_grad():
            for x, y in tqdm(trainloader):
                x = x.cpu()
                for i in range(len(x)):
                    x = x.squeeze()
                    x = x.view(x.size(0), -1)
                    new_test_data.append(x[i])
                    new_test_label.append(y[i])
        dic = {'data': new_test_data, 'label': new_test_label}
        dataframe = pd.DataFrame(dic)
        testset = DataframeToDataset.DataframetoDataset(dataframe)
        trainset = CustomedSubset(testset,[i for i in range(len(testset))])
        print(len(trainset))
        return trainset


class CustomedSubset(Dataset[T_co]):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:

        self.indices = indices
        self.data = []
        self.targets = []
        self.dataset = dataset
        for i in self.indices:
            self.data.append(dataset.data[i])
            self.targets.append(dataset.labels[i])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    def __getitem__(self, idx):
        return self.dataset[idx],self.targets[idx]

    def __len__(self):
        return len(self.indices)





