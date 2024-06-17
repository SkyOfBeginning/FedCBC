import random

import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch import tensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_fn(recon_x, x, mean, log_var):
    # MEAN μ 均值，log_var σ 方差对数
    mse = torch.nn.MSELoss(reduction='sum')
    loss = mse(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (loss + 0.5*KLD) / x.size(0)

def loss_distillation_local(recon_x, x, mean, log_var,teacher_mean,teacher_logvar,alpha):
    recon_loss = loss_fn(recon_x,x,mean,log_var)
    distill_loss =  0.5 * torch.sum(teacher_logvar-log_var-1 + (log_var.exp()+(mean-teacher_mean).pow(2))/teacher_logvar.exp())
    return recon_loss+ alpha*(distill_loss/x.size(0))

def global_distillation_loss(recon_x, x, mean, log_var,zs,lvs,alpha):
    recon_loss = loss_fn(recon_x, x, mean, log_var)
    teacher_mean = sum(zs) / len(zs)
    teacher_logvar = sum(lvs) / len(lvs)
    distill_loss = 0.5 * torch.sum(teacher_logvar - log_var - 1 + (log_var.exp() + (mean - teacher_mean).pow(2)) / teacher_logvar.exp())
    return recon_loss + alpha * (distill_loss / x.size(0))

def prediction(models,testloader,device,feature_extractor):
    feature_extractor.to(device)
    for i in models.keys():
        models[i].to(device)
        models[i].eval()

    correct, total = 0, 0
    loss = torch.nn.MSELoss()

    for x,labels in testloader:
        x= x.to(device)
        predicts = []
        # j 是 特征
        for q,j in enumerate(x):
            max_score=9999
            predict =None
            j = j.unsqueeze(0)
            with torch.no_grad():
                new_j,_ = feature_extractor(j)
            # print(new_j.shape)
            for label in models.keys():
                with torch.no_grad():
                    recon_x,mean,logvar,_ = models[label](new_j)
                score = loss_fn(recon_x,new_j,mean,logvar)
                # print(score)
                if score<max_score:
                    predict = label
                    max_score = score
            predicts.append(predict)
        # print(predicts)
        # print(labels)
        # print('-------------------------------')
        correct += (tensor(predicts).cpu() == labels.cpu()).sum()
        total += len(labels)
    for i in models.keys():
        models[i].train()
    return correct/total *100

def prediction_withS(models,testloader,device,feature_extractor):
    for i in models.keys():
        models[i].eval()
    correct, total = 0, 0
    loss = torch.nn.MSELoss()
    if feature_extractor != None:
        feature_extractor.to(device)
    for index,x,labels in testloader:
        x= x.to(device)
        if feature_extractor != None:
            with torch.no_grad():
                x, _ = feature_extractor(x)
                x = x.squeeze()
        predicts = []
        for q, j in enumerate(x):
            max_score = 0
            predict = None
            for label in models.keys():
                with torch.no_grad():
                    score = models[label].estimate_loglikelihood_single(j)
                    print(score)
                if score > max_score:
                    predict = label
                    max_score = score
            predicts.append(predict)
        correct += (tensor(predicts).cpu() == labels.cpu()).sum()
        total += len(labels)
    for i in models.keys():
        models[i].train()
    return correct / total * 100

def predict(x,models):
    score = 9999
    max =99
    predicted = 0
    loss = torch.nn.MSELoss()
    with torch.no_grad():
        for key in models.keys():
            recon, mean, log_var, z = models[key](x)
            score = loss(x,recon)
            # print(f'{key}的loss{score}')
            if score<max:
                max = score
                predicted = key
        # print(f'最后的结果是{predicted}')

    return predicted




