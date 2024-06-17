import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src import DataframeToDataset
from src.client import Client
from src.models.VAE_PRETRAINED import VAE
from src.utils import global_distillation_loss, prediction


class Server(object):
    def __init__(self,client_num,task_num,dataset,lr,device,epoch_local,pseudo_samples,global_method,local_method,
                 FE,subset,batch_size,alpha,test_dataset,class_num,mask):
        self.client_num = client_num
        self.task_num = task_num

        self.feature_extractor = FE

        self.data= subset
        self.mask =mask
        self.dataset_name = dataset
        self.test_dataset = test_dataset

        self.learning_rate = lr
        self.epoch_local = epoch_local
        self.batch_size = batch_size

        self.global_method = global_method
        self.local_method = local_method

        self.pseudo_samples = pseudo_samples
        self.device = device
        self.class_num = class_num

        self.clients = []
        self.alpha = alpha
        self.global_models={}
        self.encoder_layer_sizes = [512, 256]
        self.laten_size = 128
        self.decoder_layer_sizes = [256, 512]

        self.test_loader = test_dataset

    def init_clients(self):
        print("Init clients...")
        for i in range(self.client_num):
            self.clients.append(Client(self.dataset_name,self.learning_rate,self.device,self.task_num,i,
                                       self.epoch_local,self.pseudo_samples,self.local_method,self.feature_extractor,
                                       self.data[i],self.batch_size,self.alpha,self.mask[i]
                                       ))
        print("Complete")

    # 输入的是同一个label的model
    def FedAvg(self,models):
        model_parameters = []
        for model in models:
            para = model.state_dict()
            model_parameters.append(para)
        w_avg = copy.deepcopy(model_parameters[0])
        for k in w_avg.keys():
            for i in range(1, len(model_parameters)):
                w_avg[k] += model_parameters[i][k]
            w_avg[k] = torch.div(w_avg[k], len(model_parameters))
        return w_avg

    def fedavg_for_all(self):
        self.seen_class = set()
        # 添加class,只对这一轮里
        for i in range(self.client_num):
            for label in self.clients[i].current_class:
                self.seen_class.add(label)
        for label in self.seen_class:
            local_vae = []
            for i in range(self.client_num):
                if label in self.clients[i].models.keys():
                    local_vae.append(self.clients[i].models[label])
            global_para = self.FedAvg(local_vae)
            self.global_models[label] = VAE(self.encoder_layer_sizes,self.laten_size,self.decoder_layer_sizes)
            self.global_models[label].load_state_dict(global_para)
            self.global_models[label].to(self.device)



    def RehearsalandDistillation(self):
        # 每次蒸馏都要置空
        self.seen_class = set()
        self.models = {}
        # 添加class,只对这一轮里
        for i in range(self.client_num):
            for label in self.clients[i].current_class:
                self.seen_class.add(label)

        # 只对当前回合的，在client本地端训练过的类
        for label in tqdm(self.seen_class):
            self.models[label] = VAE(self.encoder_layer_sizes, self.laten_size, self.decoder_layer_sizes)
            local_vae =[]
            for i in range(self.client_num):
                if label in self.clients[i].models.keys():
                    local_vae.append(self.clients[i].models[label])
            # 如果local vae 只有一个，而且没有global model
            # 就直接拿过去当全局模型
            if len(local_vae)==1 and label not in self.global_models.keys():
                self.global_models[label] = copy.deepcopy(local_vae[0])

            else:
            # 上一回合的global_model，如果有就加入一起生成伪样本
                if label in self.global_models.keys():
                    local_vae.append(self.global_models[label])
                    self.models[label] = copy.deepcopy(self.global_models[label])
                else: self.models[label] = copy.deepcopy(random.choice(local_vae))

                train_loader = self.generate_samples(local_vae)
                self.models[label].to(self.device)
                optimizer = torch.optim.Adam(self.models[label].parameters(), lr=self.learning_rate,weight_decay=1e-03)
                for epoch in range(self.epoch_local):
                    for x in train_loader:
                        x = x.to(self.device)
                        recon_x, mean, log_var, z = self.models[label](x)
                        log_vars = []
                        zs = []
                        with torch.no_grad():
                            for localmodel in local_vae:
                                _,_,lv,tz = localmodel(x)
                                log_vars.append(lv)
                                zs.append(tz)
                        loss = global_distillation_loss(recon_x, x, mean, log_var,zs,log_vars,self.alpha)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                self.global_models[label] = copy.deepcopy(self.models[label])

    def generate_samples(self,localvaes):
        samples = []
        for i in range(self.pseudo_samples):
            noise = torch.tensor(np.random.uniform(-1, 1, [1, self.laten_size]).astype(np.float32), dtype=torch.float32)
            noise = noise.to(self.device)
            for j in localvaes:
                with torch.no_grad():
                    sample = j.generate(noise)
                samples.append(sample)

        trainloader = DataLoader(samples,batch_size=self.batch_size,shuffle=True)
        return trainloader


    def train_clients(self):
        for i in range(self.task_num*5):
            print(f"--------global round {i},task number {i//5}-----------")
            for j in range(self.client_num):
                self.clients[j].train(i)
            if self.global_method=="distillation":
                print("using Selective Knowledge Fusion for aggregation")
                self.RehearsalandDistillation()
            elif self.global_method=='fedavg':
                print("使用fedavg进行aggregation")
                self.fedavg_for_all()

            if (i+1) %5 ==0:
                self.model_global_eval(i)
            for j in range(self.client_num):
                self.clients[j].recive_global_models(self.global_models)

        print("All process compelete")




    def model_global_eval(self,taskid):
        print(self.global_models.keys())

        acc = prediction(self.global_models,self.test_dataset,self.device,self.feature_extractor)
        print(f'Round {taskid} finishes, the global models on the global testset acc is {acc}')





