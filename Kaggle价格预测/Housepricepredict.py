import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import d2lzh_pytorch as d2l

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
 #将trian与test的特征data连接起来，先预处理一下数据:

'''我们对连续数值的特征做标准化（standardization）：设该特征在整个数据集上的均值为μ
μ，标准差为σ
σ。那么，我们可以将该特征的每个值先减去μ
μ再除以σ
σ得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。'''
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index #返回类型不为object列的列名。
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)



'''接下来将离散数值转成指示特征。举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，
那么这一步转换将去掉MSZoning特征，
并新加两个特征MSZoning_RL和MSZoning_RM，
其值为0或1。如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。'''
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape # (2919, 331)
all_features=all_features.astype(float)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)


loss = torch.nn.MSELoss()

def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net

def log_rmse(net, features, labels): #比赛时的评估模型准测
    with torch.no_grad(): #这种方法特别适用于不需要梯度的代码段，因为它能够减少内存消耗并提高运行效率。
        # 将小于1的值设成1，使得取对数时数值更稳定
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
'''折交叉验证。它将被用来选择模型设计并调节超参数。下面实现了一个函数，
它返回第i折交叉验证时所需要的训练和验证数据。'''

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    '''
assert k > 1 是 Python 中的断言语句。它用于在代码中检查一个条件是否为真。如果条件为真（即满足条件），
则程序会继续执行；如果条件为假（即不满足条件），则会触发 AssertionError 异常，并且程序会停止执行。'''
    fold_size = X.shape[0] // k #每个fold的样本量
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) #my_slice = slice(1, 5, 2)  # 从索引 1 到索引 5，间隔为 2
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part #第i折作为验证集
        elif X_train is None: #初始填入训练集
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #除了第i折 其它折均作为训练集
            y_train = torch.cat((y_train, y_part), dim=0)
            '''torch.cat沿着指定的维度拼接张量（tensor）的函数。
            它能够将多个张量沿着给定的维度连接起来，生成一个新的张量'''
    return X_train, y_train, X_valid, y_valid

'''在K
K折交叉验证中我们训练K
K次并返回训练和验证的平均误差。'''
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        #print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


#我们使用一组未经调优的超参数并计算交叉验证误差。可以改动这些超参数来尽可能减小平均测试误差。
#所以k-fold写了半天只是为了选择超参数？
k, num_epochs, lr, weight_decay, batch_size = 4, 150, 14.5, 0, 64

#那我们先来选一选lr:
def find_best_lr(find_epoch,start_lr,rate):
   valid_rate = 1
   best_lr = start_lr
   lr=start_lr
   for i in range(20):
      train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
      print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
      if valid_rate>=valid_l:
        best_lr=lr
        valid_rate=valid_l
      lr+=rate
   return best_lr


#best_lr=find_best_lr(50,start_lr=14,rate=5)
a,b=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
print(b)
best_lr=lr

def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, best_lr, weight_decay, batch_size)

