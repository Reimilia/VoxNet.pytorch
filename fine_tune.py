#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: train.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description:
'''

from __future__ import print_function
import argparse
import sys
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from voxnet import VoxNet
from transformer import NaiveTransformer,FeatureTransformer,TokenizedTransformer,FPNTransformer
from vit_3d_transformer import ViT3D,ViT3D_Efficient
sys.path.insert(0, './data/')
from modelnet10 import ModelNet10
from modelnet40 import ModelNet40
from shapenet_v2 import ShapeNetV2
from linformer import Linformer
from collections import OrderedDict

efficient_transformer = Linformer(
    dim = 32,
    seq_len = 32*32*32 + 1,  # 64 x 64 patches + 1 cls token
    depth = 4,
    heads = 8,
    k = 128
)


CLASSES_ModelNet10 = {
    0: 'bathtub',
    1: 'chair',
    2: 'dresser',
    3: 'night_stand',
    4: 'sofa',
    5: 'toilet',
    6: 'bed',
    7: 'desk',
    8: 'monitor',
    9: 'table'
}

CLASSES_ModelNet40 = {
        0:'airplane', 1:'bathtub', 2:'bed', 3:'bench',
        4:'bookshelf', 5:'bottle', 6:'bowl', 7:'car',
        8:'chair', 9:'cone', 10:'cup', 11:'curtain',
        12:'desk', 13:'door', 14:'dresser', 15:'flower_root',
        16:'glass_box', 17:'guitar', 18:'keyboard', 19:'lamp',
        20:'laptop', 21:'mantel', 22:'monitor', 23:'night_stand',
        24:'person', 25:'piano', 26:'plant', 27:'radio',
        28:'range_hood', 29:'sink', 30:'sofa', 31:'stairs',
        32:'stool', 33:'table', 34:'tent', 35:'toilet',
        36:'tv_stand', 37:'vase', 38:'wardrobe', 39:'xbox'
    }

CLASSES_SHAPENET = {0: '02691156', 1: '02747177', 2: '02773838', 3: '02801938',
               4: '02808440', 5: '02818832', 6: '02828884', 7: '02843684',
               8: '02871439', 9: '02876657',10: '02880940',11: '02924116',
               12:'02933112',13: '02942699',14: '02946921',15: '02954340',
               16:'02958343',17: '02992529',18: '03001627',19: '03046257',
               20:'03085013',21: '03207941',22: '03211117',23: '03261776',
               24:'03325088',25: '03337140',26: '03467517',27: '03513137',
               28:'03593526',29: '03624134',30: '03636649',31: '03642806',
               32:'03691459',33: '03710193',34: '03759954',35: '03761084',
               36:'03790512',37:'03797390',38:'03928116',39:'03938244',
               40:'03948459',41:'03991062',42:'04004475',43:'04074963',
               44:'04090263',45:'04099429',46:'04225987',47:'04256520',
               48:'04330267',49:'04379243',50:'04401088',51:'04460130',
               52:'04468005',53:'04530566',54:'04554684'
               }

def blue(x): return '\033[94m' + x + '\033[0m'

# 参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, default='./data/ModelNet10', help="dataset path")
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--n-epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='./cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default='ShapeNetV2', help='which dataset to be used')
parser.add_argument('--gpu', type=int, default=1 , help='which GPU to use')
opt = parser.parse_args()
# print(opt)
opt.dataset = 'ModelNet10'

downsample= False
opt.outf='./cls/fine_tune_256'
opt.model ='./cls/cls_model_49.pth'


if torch.cuda.is_available():
    device = torch.device("cuda:%d" % opt.gpu)
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# 创建目录
try:
    os.makedirs(opt.outf)
except OSError:
    pass

# 固定随机种子
opt.manualSeed = 9
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# 数据加载
if opt.dataset == 'ModelNet10':
    opt.data_root = './data/ModelNet10'
    CLASSES= CLASSES_ModelNet10
    N_CLASSES=len(CLASSES)
    train_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
    test_dataset = ModelNet10(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')
elif opt.dataset == 'ShapeNetV2':
    opt.data_root = '/mnt/storage/datasets/ShapeNetCore_v2'
    CLASSES= CLASSES_SHAPENET
    N_CLASSES=len(CLASSES)
    dataset = ShapeNetV2(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES)
    print("There are %d models in ShapeNetCoreV2" % len(dataset))
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(9))
elif opt.dataset == 'ModelNet40':
    opt.data_root = '/mnt/storage/yiwang/data/ModelNet40_Aligned'
    CLASSES= CLASSES_ModelNet40
    N_CLASSES=len(CLASSES)
    train_dataset = ModelNet40(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='train')
    test_dataset = ModelNet40(data_root=opt.data_root, n_classes=N_CLASSES, idx2cls=CLASSES, split='test')
else:
    pass



train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

# VoxNet
#voxnet = VoxNet(n_classes=N_CLASSES)
#voxnet = TokenizedTransformer(n_classes=N_CLASSES)
voxnet = FeatureTransformer(n_classes=len(CLASSES_SHAPENET), input_shape=(32, 32, 32))
#voxnet = FPNTransformer(n_classes=N_CLASSES)
#voxnet = ViT3D(voxel_size = 32, dim = 256, patch_size = 4, depth = 6, heads = 8, mlp_dim = 1024, num_classes=N_CLASSES, channels=1, dropout=0.3)
#voxnet = ViT3D_Efficient(voxel_size = 32, dim = 32, patch_size = 1, num_classes = N_CLASSES, transformer=efficient_transformer, channels=1)



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            if name=='mlp.fc2.weight' or name=='mlp.fc2.bias':
                continue
            #if name=='mlp.fc1.weight' or name=='mlp.fc1.bias':
            #    continue
            param.requires_grad = False






# 加载权重
if opt.model != '':
    voxnet.load_state_dict(torch.load(opt.model))
    voxnet.mlp[-1] = torch.nn.Linear(256, N_CLASSES)
    set_parameter_requires_grad(voxnet, True)
print(voxnet)

# 优化器
optimizer = optim.Adam(voxnet.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     voxnet = torch.nn.DataParallel(voxnet)

voxnet.to(device)

num_batch = len(train_dataset) / opt.batchSize
print(len(train_dataset))

for epoch in range(opt.n_epoch):
    # scheduler.step()
    for i, sample in tqdm(enumerate(train_dataloader, 0)):
        # 读数据
        voxel, cls_idx = sample['voxel'], sample['cls_idx']
        voxel, cls_idx = voxel.to(device), cls_idx.to(device)
        voxel = voxel.float()  # Voxel原来是int类型(0,1),需转float, torch.Size([256, 1, 32, 32, 32])
        if downsample:
            voxel = torch.nn.functional.interpolate(voxel, size=(32,32,32), mode="trilinear")

        # 梯度清零
        optimizer.zero_grad()

        # 网络切换训练模型
        voxnet = voxnet.train()
        pred = voxnet(voxel)  # torch.Size([256, 10])
        # 计算损失函数

        loss = F.cross_entropy(pred, cls_idx)

        # 反向传播, 更新权重
        loss.backward()
        optimizer.step()

        # 计算该batch的预测准确率
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(cls_idx.data).cpu().sum()
        #print('[%d: %d/%d] train loss: %f accuracy: %f' %
        #      (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        total_correct = 0
        total_testset = 0


        # 每5个batch进行一次test
        # if i % 5 == 0:
        #     j, sample = next(enumerate(test_dataloader, 0))
        #     voxel, cls_idx = sample['voxel'], sample['cls_idx']
        #     voxel, cls_idx = voxel.to(device), cls_idx.to(device)
        #     voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])
        #     voxnet = voxnet.eval()
        #     pred = voxnet(voxel)
        #     loss = F.nll_loss(pred, cls_idx)
        #     pred_choice = pred.data.max(1)[1]
        #     correct = pred_choice.eq(cls_idx.data).cpu().sum()
        #     print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch,
        #                                                     blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
    for i, data in tqdm(enumerate(test_dataloader, 0)):
        voxel, cls_idx = data['voxel'], data['cls_idx']
        voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
        voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])
        if downsample:
            voxel = torch.nn.functional.interpolate(voxel, size=(32,32,32), mode="trilinear")
        voxnet = voxnet.eval()
        pred = voxnet(voxel)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(cls_idx.data).cpu().sum()
        total_correct += correct.item()
        total_testset += voxel.size()[0]

    print("Epoch %d test accuracy %f" % (epoch, total_correct / float(total_testset)))
    # 保存权重
    torch.save(voxnet.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


# 训练后, 在测试集上评估
total_correct = 0
total_testset = 0

for i, data in tqdm(enumerate(test_dataloader, 0)):
    voxel, cls_idx = data['voxel'], data['cls_idx']
    voxel, cls_idx = voxel.cuda(), cls_idx.cuda()
    voxel = voxel.float()  # 转float, torch.Size([256, 1, 32, 32, 32])
    if downsample:
        voxel = torch.nn.functional.interpolate(voxel, size=(32,32,32), mode="trilinear")
    voxnet = voxnet.eval()
    pred = voxnet(voxel)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(cls_idx.data).cpu().sum()
    total_correct += correct.item()
    total_testset += voxel.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))
