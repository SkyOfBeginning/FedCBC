import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def accuracy(output, target, topk=(1,)):
    ### 返回的是正确的个数
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:min(k, maxk)].reshape(-1).float().sum(0)  for k in topk],batch_size


def global_distillation_loss(output,outputs):
    mse = torch.nn.MSELoss(reduction='sum')
    total_loss = 0
    for i in outputs:
        loss=mse(output,i)
        total_loss +=loss
    return total_loss

def build_transform(is_train,input_size):
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop([input_size,input_size], scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    return transforms.Compose(t)

def visualize_feature_map(img_batch,name):
    feature_map = img_batch[0]
    print(feature_map.shape) # 1 64 7 7
    feature_map = feature_map.reshape(feature_map.shape[1],feature_map.shape[2],-1)
    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[0]
    row, col = 32,32

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :,i]
        feature_map_combination.append(feature_map_split)
        # plt.subplot(row, col, i + 1)
        # plt.imshow(feature_map_split)

    plt.savefig(name)
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")

def get_row_col( num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def euclidean_metric_cal(a, b):
    euc_metric = torch.empty(0, dtype=torch.long).cuda()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            euc = torch.sqrt(((a[i][j].repeat(b.shape[1], 1) - b[i]) ** 2).sum(dim=1))
            euc_metric = torch.cat([euc_metric, euc.unsqueeze(0)])
    return euc_metric.view(-1, a.shape[1], b.shape[1])


def function_cal(x, y, temperature, kccl_euc=0, ks=1, km=0):
    if kccl_euc == 1:
        euc_dis = euclidean_metric_cal(x, y)
        return torch.exp(torch.div(1, euc_dis * temperature))
    else:
        return torch.exp((torch.div(torch.bmm(x, y.permute(0, 2, 1)), temperature) - km) * ks)


def kccl_loss(pooled_output, labels, k, temperature, neg_num=1, weight=None, loss_metric=0, neg_method=0,
              centroids=None, kccl_euc=0, ks=1, km=0):
    features = torch.cat((pooled_output, labels.unsqueeze(1)), 1)
    B, H = pooled_output.shape
    if neg_method in [3, 6]:
        pooled_output = pooled_output.view(-1, 1 + k + neg_num, H)
        labels = labels.view(-1, 1 + k + neg_num)
        neg_examples = pooled_output[:, (1 + k):, :]
        pooled_output = pooled_output[:, :(1 + k), :]

    pos = function_cal(pooled_output, pooled_output, temperature, kccl_euc, ks, km)
    neg1 = function_cal(pooled_output, neg_examples, temperature, kccl_euc, ks, 0)
    if neg_num > 1 or neg_examples.shape[1] > 1:
        neg1 = torch.sum(neg1, dim=2).unsqueeze(2)
    neg2 = neg1.permute(0, 2, 1).repeat(1, k + 1, 1)
    pos_neg_mask = 1 - torch.eye(pos.shape[-1], device=neg2.device).unsqueeze(0).repeat(pos.shape[0], 1, 1)
    pos_neg_mask = pos_neg_mask.float()
    pos_neg = torch.sum(torch.sum(pos * pos_neg_mask, dim=-1), dim=-1).unsqueeze(1).unsqueeze(1).repeat(1,
                                                                                                        *pos.shape[1:])
    pos_neg = pos_neg - pos * 2
    neg = pos + neg2 + neg2.permute(0, 2, 1) + pos_neg
    loss_a = - torch.log(torch.div(pos, neg))
    if loss_metric == 0:
        for i in range(loss_a.shape[1]):
            loss_a[:, i, i] = 0
        loss_b = torch.sum(torch.sum(loss_a, dim=1))
        loss = loss_b / (pooled_output.shape[0] * k * (k + 1))
    elif loss_metric == 1:
        for i in range(loss_a.shape[1]):
            for j in range(loss_a.shape[2]):
                if i >= j:
                    loss_a[:, i, j] = 0
        loss_b = torch.sum(torch.sum(loss_a, dim=1))
        loss = 2 * loss_b / (pooled_output.shape[0] * k * (k + 1))
    return loss

def CosineSimilarityClassifier(test_feature,global_protos,current_class):
    max_similarity = -1
    predicted_label = None

    all_global_protos_keys = np.array(list(current_class))
    all_protos = []
    for protos_key in all_global_protos_keys:
        all_protos.append(global_protos[protos_key])
    all_protos = torch.tensor(np.vstack(all_protos)).to('cuda')

    l = torch.cosine_similarity(test_feature, all_protos, dim=1)
    print(l)

    _,index = torch.topk(l,dim=0,k=1)

    predicted_label = all_global_protos_keys[index]


    return predicted_label
