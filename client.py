import copy
import random
import time
from copy import deepcopy

import numpy as np
import pandas
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from src.DataframeToDataset import ListtoDataset
from src.utils import loss_distillation_local, loss_fn, prediction, prediction_withS
from src.models.VAE_PRETRAINED import VAE


class Client(object):
    def __init__(self,dataset,lr,device,task_num,id,epoch_local,pseudo_samples,local_method,FE,subset,batch_size,alpha,mask):
        self.id = id
        self.task_num = task_num
        self.learning_rate = lr
        self.pseudo_samples = pseudo_samples
        self.epoch_local = epoch_local
        self.batch_size = batch_size
        self.alpha = alpha

        self.device = device

        self.dataset = dataset
        self.mask = mask

        self.feature_extractor = FE
        self.local_method = local_method

        self.test_loader = []
        self.train_dataset = subset
        self.train_data = None

        self.models = {}
        self.global_models={}
        self.task_id =-1

        # self.encoder_layer_sizes = [4096,1024,512]
        self.encoder_layer_sizes = [512,256]
        self.laten_size = 128
        self.decoder_layer_sizes = [256,512]
        self.first_task = None

    def get_data(self,task_id):
        # CIFAR10 & MNIST

        if task_id%5 == 0:
            self.task_id = task_id//5
            self.current_class = self.mask[self.task_id]
            trainset = self.train_dataset[self.task_id]
            if self.task_id == 0:
                self.first_task = self.current_class


        print(f'{self.id} client，{task_id} task contains {self.current_class} classes, there are {len(self.current_class)} classes')


        traindata, testdata = random_split(trainset,
                                           [int(len(trainset) * 0.8), len(trainset) - int(len(trainset) * 0.8)])
        testdata = deepcopy(testdata)
        testloader = DataLoader(testdata,shuffle=True,batch_size=self.batch_size)

        self.test_loader.append(testloader)
        self.train_data = traindata
        # 修改


    def data_feature_extract_process(self):
        # print(f"{self.id}号客户端特征提取中")
        self.processed_testdata = {}
        with torch.no_grad():
            self.feature_extractor.to(self.device)
            for x,label in self.train_data:
                x = x.to(self.device)
                x = x.unsqueeze(0)
                if label not in self.processed_testdata.keys():
                    self.processed_testdata[label] = []
                    newx,_ = self.feature_extractor(x)
                    newx = newx.squeeze()
                    self.processed_testdata[label].append(newx)
                else:
                    newx,_ = self.feature_extractor(x)
                    newx = newx.squeeze()
                    self.processed_testdata[label].append(newx)
        # print(f"{self.id}号客户端特征提取完毕")


    def generate_pseudo_samples(self):
        for label in self.processed_testdata.keys():
            if label in self.global_models.keys() and label in self.models.keys():
                for i in range(self.pseudo_samples):
                    with torch.no_grad():
                        self.global_models[label].to("cpu")
                        noise = torch.tensor(np.random.uniform(-1, 1, [1, self.laten_size]).astype(np.float32), dtype=torch.float32)
                        noise = noise.to("cpu")
                        n = self.global_models[label].generate(noise)
                        n = n.squeeze().to('cuda')
                    self.processed_testdata[label].append(n)
            # 当前数据集处理的label local——model有，global——没得（接收的时候把global去掉了）
            # 用本地模型生成伪样本
            elif label in self.models.keys() and label not in self.global_models.keys():
                for i in range(self.pseudo_samples):
                    with torch.no_grad():
                        self.models[label].to("cpu")
                        noise = torch.tensor(np.random.uniform(-1, 1, [1, self.laten_size]).astype(np.float32),
                                             dtype=torch.float32)
                        noise = noise.to("cpu")
                        n = self.models[label].generate(noise)
                        n = n.squeeze().to('cuda')
                    self.processed_testdata[label].append(n)


    def recive_global_models(self,global_models:dict):
        self.global_models = copy.deepcopy(global_models)
        # local不蒸馏，全部用globalmodel替换
        # self.evaluate_global_models(self.task_id)
        if self.local_method =="replace":
            for i in self.global_models.keys():
                self.models[i] = deepcopy(self.global_models[i])
            self.global_models.clear()
        else:
            used_key = []
            for i in self.global_models.keys():
                # 本地里没有的，就直接用global的代替本地的
                if i not in self.models.keys():
                    self.models[i] = deepcopy(self.global_models[i])
                    used_key.append(i)
            for i in used_key:
                self.global_models.pop(i)



    def train(self,task_id):
        task = task_id//5
        if self.task_id != task:
            self.get_data(task_id)
            self.task_id=task
        if self.feature_extractor != None:
            self.data_feature_extract_process()
        # self.processed_testdata = {}
        # for i, x in self.train_data:
        #     x = int(x)
        #     if x not in self.processed_testdata.keys():
        #         self.processed_testdata[x] = []
        #         self.processed_testdata[x].append(i.squeeze())
        #     self.processed_testdata[x].append(i.squeeze())

        self.train_loader = {}

        if task_id != 0:
            if self.local_method == 'distillation':
                self.generate_pseudo_samples()  # generate pseudo samples into train dataset

        for i in self.processed_testdata.keys():
            self.train_loader[i] = DataLoader(self.processed_testdata[i], batch_size=self.batch_size, shuffle=True)

        # 开始训练
        print(f'{self.id} client is training on No. {task_id} task')
        for label in tqdm(self.train_loader.keys()):
            # 判断local有没有，global有没有
            # local有，global有
            if self.local_method=="distillation":
                if label in self.models.keys() and label in self.global_models.keys():
                    # distillation
                    self.models[label].to(self.device)
                    self.global_models[label].to(self.device)
                    optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                    mse = torch.nn.MSELoss(reduction='sum')
                    for epoch in range(self.epoch_local):
                        for iteration, (x) in enumerate(self.train_loader[label]):
                            x = x.to(self.device)
                            recon_x, mean, log_var, z = self.models[label](x)
                            with torch.no_grad():
                                teacher_x, teacher_mean, teacher_log_var, teacher_z = self.global_models[label](x)
                            loss = loss_distillation_local(recon_x,x,mean,log_var,teacher_mean,teacher_log_var,self.alpha)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                else:
                # local无。global无
                    if label not in self.models.keys():
                        self.models[label] = VAE(self.encoder_layer_sizes,self.laten_size,self.decoder_layer_sizes)   # 新增一个
                # local有，global无
                    self.models[label].to(self.device)
                    optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                    mse = torch.nn.MSELoss(reduction='sum')
                    for epoch in range(self.epoch_local):
                        for iteration, (x) in enumerate(self.train_loader[label]):
                            x = x.to(self.device)
                            # print(x.shape)
                            recon_x, mean, log_var, z = self.models[label](x)

                            loss = mse(recon_x, x)
                            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                            loss1 = (loss + 0.5*KLD) / x.size(0)
                            # print(loss)
                            optimizer.zero_grad()
                            loss1.backward()
                            optimizer.step()
            else:
                # local_method is 'replace', which means global models replace local models

                if label not in self.models.keys():
                    self.models[label] = VAE(self.encoder_layer_sizes,self.laten_size,self.decoder_layer_sizes)   # 新增一个
                self.models[label].to(self.device)
                optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                for epoch in range(self.epoch_local):
                    for iteration, (x) in enumerate(self.train_loader[label]):
                        x = x.to(self.device)
                        recon_x, mean, log_var, z = self.models[label](x)
                        loss = loss_fn(recon_x,x,mean,log_var)
                        # print(loss)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

        print(f'{self.id} client finish training')

        self.evaluate_local_models(0)
        self.evaluate_local_models(task_id)



    def evaluate_global_models(self,index):
        models = {}
        for i in self.current_class:
            models[i] = self.global_models[i]
        acc = prediction(models,self.test_loader[index],self.device)
        print(f'global models 在{self.id}号客户端的{index}号测试集的正确率为{acc}')

    def evaluate_local_models(self,index):
        index = index // 5
        models ={}
        for i in self.current_class:
            models[i] = self.models[i]

        acc = prediction(models, self.test_loader[index],self.device,self.feature_extractor)
        print(f'Client {self.id}\'s local models on {index} task acc is {acc}')






































