import time
import argparse
import numpy as np
import torch.cuda

import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *

parser = argparse.ArgumentParser()
# 使用argparse的第一步是创建一个ArgumentParser对象。
# ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息。
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA training.')
# 通过调用add_argument()来给一个ArgumentParser添加程序参数信息。
# 第一个参数 - 选项字符串，用于作为标识
# action - 当参数在命令行中出现时使用的动作基本类型
# default - 当参数未在命令行中出现时使用的值
# type - 命令行参数应当被转换成的类型
# help - 一个此选项作用的简单描述
# 此句是 禁用CUDA训练
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')

parser.add_argument('--seed', type=int, default=42, help='Random seed')

parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay(L2 loss on parameters).')
# 权重衰减
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--K', type=int, default=3,
                    help='Chebyshev order')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
# 产生随机种子，以使得结果是确定的
torch.manual_seed(args.seed)
# 为CPU设置随机种子用于生成随机数，以使得结果是确定的

adj, features, labels, idx_train, idx_val, idx_test = load_data()

model = myGCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout,
              adj=adj,
              K=args.K)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    # 模型放到GPU上跑
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    # 把这些参数全部放到GPU上运算


def train(epoch):
    t = time.time()
    model.train()
    # 固定语句，主要针对启用BatchNormalization和Dropout
    optimizer.zero_grad()

    output = model(features)

    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()

    optimizer.step()

    if not args.fastmode:
        model.eval()
        # 固定语句，主要针对不启用BatchNormalization和Dropout
        output = model(features)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])

        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch + 1),  # 正在迭代的epoch数
              'loss_train: {:.4f}'.format(loss_train.item()),  # 训练集损失函数值
              'acc_train: {:.4f}'.format(acc_train.item()),  # 训练集准确率
              'loss_val: {:.4f}'.format(loss_val.item()),  # 验证集损失函数值
              'acc_val: {:.4f}'.format(acc_val.item()),  # 验证集准确率
              'time: {:.4f}s'.format(time.time() - t))  # 运行时间


def test():
    model.eval()
    output = model(features)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])

    acc_test = accuracy(output[idx_test], labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),  # 测试集损失函数值
          "accuracy= {:.4f}".format(acc_test.item()))  # 测试集的准确率


t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

print("Optimization Finished!")  # 优化完成！
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))  # 已用总时间

# 测试
test()
