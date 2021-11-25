import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from scipy.sparse.linalg import eigsh

def encode_onehot(labels):
    """
    # 将所有的标签整合成一个one-hot编码，这样就不会重复
    classes = set(labels) # set() 函数创建一个无序的不重复元素集，删除重复，留下不同
    # 运行前 labels有2708个值(2708篇论文的类别) 运行后，classes只有7个值(即7个类别)
    '''enumerate()函数生成序列，带有索引i和值c。
    这一句将string类型的label变为int类型的label，建立映射关系
    np.identity(len(classes))为创建一个classes的单位矩阵
    创建一个字典，索引为label，值为独热码向量'''
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    # enumerate(classes) 返回索引i和值c。
    # np.identity(len(classes))[i, :]创建单位方阵，维度为len(classes)，
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    """
    """省事做法"""
    labels_onehot = np.array(pd.get_dummies(labels,dummy_na=False), dtype=np.int32)
    # 直接独热编码get_dummies，如何返回
    return labels_onehot

def normalize(mx):
    # 归一化矩阵
    rowsum = np.array(mx.sum(1), dtype=np.float32) # mx.sum(1)对每一行求和
    r_inv = np.power(rowsum, -1).flatten() # 得到倒数
    # 计算倒数有个问题，如果数为0，则倒数为无穷大，使用isinf看哪个最大，然后改成0
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """稀疏矩阵转化成张量函数"""
    """
    先转换成COO格式的稀疏矩阵，再把稀疏矩阵转换成张量的形式输出，便于后续的乘法运算
    numpy中的ndarray转化成pytorch中的tensor : torch.from_numpy()
    pytorch中的tensor转化成numpy中的ndarray : numpy()
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # vstack 垂直方向堆叠数组，from_numpy数组转化成张量
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def accuracy(output, labels):
    # 使用type_as(tensor)将张量转化为给定类型的张量
    preds = output.max(1)[1].type_as(labels)
    # output.max(1) 每一行取最大值
    # max的结构是[0]：values [1]：indices 最大值所在位置索引
    # 记录等于preds的label eq:equal
    correct = preds.eq(labels).double()
    # eq 是判断preds与labels是否相等，相等的话对应元素置1，不等置0
    correct = correct.sum() # 求和
    return correct / len(labels) # 计算准确率

def load_data(path="../cora/", dataset="cora"):
    print('loading {} dataset...'.format(dataset))
    # print('loading {dataset} dataset...')
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 读取文件并生成一个数组，数据类型为string
    # 读的cora.content
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 读取样本的特征，也就是1433维
    # idx_features_labels[:, 1:-1]只读取第二列到倒数第二列
    # 之后用来做输入
    # csr_matrix稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])
    # idx_features_labels[:, -1]读取最后一列
    # 然后转到encode_onehot来进行one-hot编码形式

    # 构建图
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 提取论文编号
    idx_map = {j: i for i, j in enumerate(idx)}
    # {}生成字典，论文编号按顺序作为key， i = 0，1，2...作为value
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 论文样本之间的引用关系的数组
    # np.genfromtxt()函数用于从.csv文件或.tsv文件中生成数组
    # np.genfromtxt(fname, dtype)
    # frame：文件名	../data/cora/cora.cites		dtype：数据类型	int32

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 将论文样本之间的引用关系用样本字典索引之间的关系表示
    # idx_map.get(edges_unordered里的数) = key对应的值，设为i，如35->i
    # idx_map.get(edges_unordered里的数) = key对应的值，设为j，如1033->j
    # 再转成edges_unordered这样n行2列数组，那么edges里的就是[i,j],...
    # 论文编号是乱的，转换到0-2707

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0],edges[:, 1]))
                        ,shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # coo_matrix((data,(row,col)))
    """    
    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])
    coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])
    """
    # 建立对称邻接矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 归一化
    features = normalize(features)
    # A + I ，对adj的归一化
    # 因为要使用myGraphConvolution
    #adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(200) # 0~139, 训练集索引列表
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    # range()创建整数列表

    features = torch.FloatTensor(np.array(features.todense()))  # 将特征矩阵转化成张量形式
    # .todense()与.csr_matrix()对应，将压缩的稀疏矩阵进行还原
    labels = torch.LongTensor(np.where(labels)[1])
    # np.where(condition)，输出满足条件condition(非0)的元素的坐标，np.where()[1]则表示返回列的索引、下标值
    # 说白了就是将每个标签one-hot向量中非0元素位置输出成标签
    # one-hot向量label转常规label：0,1,2,3,……

    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 将scipy稀疏矩阵转换为torch稀疏张量，具体函数下面有定义,因为使用的myGCN,这条也要注释掉

    idx_train = torch.LongTensor(idx_train)  # 训练集索引列表
    idx_val = torch.LongTensor(idx_val)  # 验证集索引列表
    idx_test = torch.LongTensor(idx_test)  # 测试集索引列表
    # 转化为张量
    return adj, features, labels, idx_train, idx_val, idx_test
# 返回（样本关系的对称邻接矩阵的稀疏张量，样本特征张量，样本标签(0-7的数字)，
#		训练集索引列表，验证集索引列表，测试集索引列表）

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

"""以下是切比雪夫gcn要用到的"""
def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)  # D^{-1/2}AD^{1/2}
    laplacian = sp.eye(adj.shape[0]) - adj_normalized  # L = I_N - D^{-1/2}AD^{1/2}
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')  # \lambda_{max}
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])  # 2/\lambda_{max}L-I_N

    # 将切比雪夫多项式的 T_0(x) = 1和 T_1(x) = x 项加入到t_k中
    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    # 依据公式 T_n(x) = 2xT_n(x) - T_{n-1}(x) 构造递归程序，计算T_2 -> T_k项目
    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k + 1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            return sparse_mx_to_torch_sparse_tensor(mx)

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx