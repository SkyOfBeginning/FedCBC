import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from option import args_parser
from src.ResNet import resnet18_cbam
from src.Server import Server
from src.data_process.cifar100_subset_spliter import cifar100_Data_Spliter
from src.data_process.cifar10_subset_spliter import cifar10_Data_Spliter
from src.data_process.iCIFAR10 import iCIFAR10, iCIFAR10test
from src.data_process.iCIFAR100c import iCIFAR100c
from src.data_process.iMNIST import iMNIST
from src.data_process.mnist_subset_spliter import Mnist_Data_Spliter
from utils import setup_seed


start=time.time()
args = args_parser()

setup_seed(args.seed)

feature_extractor = None
if args.pretrained_model:
    feature_extractor = torch.load('./pretrain_models/resnet-forCIFAR100-30.pth')
    # pretrained_state_dict = torch.load('./pretrain_models/resnet-forCIFAR100-0.0001.pth')
    # feature_extractor.load_state_dict(pretrained_state_dict)


train_transform =transforms.Compose([transforms.ToTensor()
                                                    , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
                                                ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_dataset = []
test_dataset = None
if args.dataset == 'CIFAR100':
    client_data, client_mask = cifar100_Data_Spliter(client_num=args.num_clients, task_num=10,
                                                     private_class_num=25,
                                                     input_size=args.img_size).random_split()
    surro_data, test_data = cifar100_Data_Spliter(client_num=args.num_clients, task_num=args.tasks_global,
                                                  private_class_num=25,
                                                  input_size=args.img_size).process_testdata(0)
    surro_data = iCIFAR100c(subset=surro_data)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)
    # cifar100_Data_Spliter(client_num=args.num_clients, task_num=args.tasks_global,
    #                       private_class_num=25,
    #                       input_size=args.img_size).train_feature_extractor()



elif args.dataset == 'CIFAR10':
    cifar10_subset = cifar10_Data_Spliter(args.num_clients, args.tasks_global,feature_extractor).random_split()  # 返回3个clients各自的数据集
    for i in range(len(cifar10_subset)):
        train_dataset.append(iCIFAR10(cifar10_subset[i]))
    test_dataset = cifar10_Data_Spliter(args.num_clients, args.tasks_global,feature_extractor).process_testdata()
    test_dataset = iCIFAR10(test_dataset)
elif args.dataset == 'MNIST':
    mnist_subset = Mnist_Data_Spliter(args.num_clients, args.tasks_global).random_split()  # 返回3个clients各自的数据集
    for i in tqdm(range(len(mnist_subset))):
        train_dataset.append(iMNIST(mnist_subset[i]))
    test_dataset = Mnist_Data_Spliter(args.num_clients, args.tasks_global).process_testdata()
    test_dataset = iMNIST(test_dataset)



num_clients = args.num_clients
global_method = args.global_method
dataset_name = args.dataset
local_method = args.local_method
batch_size = args.batch_size

epoch = args.epochs_local
lr = args.learning_rate
task_num = args.tasks_global
pretrain_model = args.pretrained_model

alpha = args.alpha
device = args.device
pseudo_samples = args.pseudo_samples
if dataset_name=="CIFAR100":
    class_num = 100
else:
    class_num = 10

myServer = Server(num_clients,task_num,dataset_name,lr,device,epoch,pseudo_samples,global_method,local_method
                  ,feature_extractor,client_data,batch_size,alpha,test_loader,class_num,client_mask)

myServer.init_clients()
# myServer.extract_test_data()
myServer.train_clients()
end=time.time()
print(f'time is {end-start}')


