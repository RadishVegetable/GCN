import math
import torch
from torch import nn
from utils import *
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    参数：
    	in_features：输入特征，每个输入样本的大小
    	out_features：输出特征，每个输出样本的大小
    	bias：偏置，如果设置为False，则层将不会学习加法偏差。默认值：True
    属性：
    	weight：形状模块的可学习权重（out_features x in_features）
    	bias：形状模块的可学习偏差（out_features）
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 先转化为张量，再转化为可训练的Parameter对象
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)  # 向我们建立的网络module添加parmeter 为什么不是self.bias = None
        self.reset_parameters()

    def reset_parameters(self):  # 随机初始化参数
        stdv = 1. / math.sqrt(self.weight.size(1))
        # size包括(in_features, out_features)，size(1)应该是out_features

        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        # 可以使用self.weight = nn.Parameter(torch.rand(in_features, out_features))
        # 来代替随机初始化参数

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # input 和 slef.weight矩阵相乘
        output = torch.spmm(adj, support)
        # spmm()是稀疏矩阵乘法，说白了还是乘法而已，只是减小了运算复杂度
        # 最新spmm函数移到了torch.sparse模块下，但是不能用
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):  # 打印输出
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    # 打印形式是：GraphConvolution (输入特征 -> 输出特征)


# 自己理解的gcn层
# 使用这个的时候，记得在load_data中输出未经normalize的adj
class myGraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, adj, K, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.t_k = chebyshev_polynomials(adj, K)
        self.weight = nn.Parameter(torch.rand(K, in_features, out_features))
        self.K = K
        if bias:
            self.bias = nn.Parameter(torch.rand(out_features))
        else:
            self.register_parameter('bias', None)  # 向我们建立的网络module添加parmeter 为什么不是self.bias = None?

    def forward(self, input):
        output = torch.zeros(input.shape[0], self.out_features)
        for i in range(1, self.K + 1):
            support = torch.mm(input, self.weight[i - 1])
            output = output + torch.spmm(self.t_k[i], support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):  # 打印输出
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    # 打印形式是：GraphConvolution (输入特征 -> 输出特征)
