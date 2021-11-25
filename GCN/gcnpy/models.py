import torch.nn as nn
import torch.nn.functional as F
from layers import *

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)

        self.dropout = dropout


    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

class myGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, adj, K):
        super().__init__()
        self.gc1 = myGraphConvolution(nfeat, nhid, adj, K)
        self.gc2 = myGraphConvolution(nhid, nclass, adj, K)

        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.gc1(x))

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x)

        return F.log_softmax(x, dim=1)
        # dim=1,对每一行的所有元素进行log_softmax运算
