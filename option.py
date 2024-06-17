import argparse
import torch

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR100', help="name of dataset")
    parser.add_argument('--global_method', type=str, default='distillation', help="name of method")
    parser.add_argument('--local_method', type=str, default='distillation', help="name of method")
    parser.add_argument('--numclass', type=int, default=100, help="number of data classes in the first task")
    parser.add_argument('--img_size', type=int, default=32, help="size of images")
    parser.add_argument('--device', type=str, default="cuda", help="GPU or CPU")
    parser.add_argument('--batch_size', type=int, default=32, help='size of mini-batch')

    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--pseudo_samples', type=int, default=100, help='number of pseudo samples')
    parser.add_argument('--epochs_local', type=int, default=300, help='local epochs ')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_clients', type=int, default=3, help='number of clients')
    parser.add_argument('--tasks_global', type=int, default=10, help='total number of tasks')
    parser.add_argument('--pretrained_model',type=bool,default=True,help='Is pretrain model available')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight of distillation loss')
    args = parser.parse_args()
    return args