#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
x, y都是一个buyer_number * sell_number的二维tensor,需要将x, y两个二维的tensor通过一系列的手段转换成三维的tensor，手段主要就是
tensor和numpy之间的转换，和numpy之间的合并
后话：要是之后做买卖双方人数不对等的话，这是一个方向，因为现在全部的迭代双边拍卖都是买卖双方人数对等，这个x, y合并程序需要重新设计，需要与神经网络的输入输出的维度相搭配
2020/01/01
"""
from parameter_Iter_BAP_linux import *


def combination_x_y(x, y):
    x = x.detach().numpy()
    y = y.detach().numpy()
    xy = np.array([x, y])
    xy = torch.Tensor(xy)
    return xy


def combination_x_y_max(x, y):
    xy = torch.zeros(size=[BATCH_SIZE, 2, buyer_number, seller_number])
    for batch_size_order in range(BATCH_SIZE):
        xy[batch_size_order] = combination_x_y(x[batch_size_order], y[batch_size_order])
    return xy


def test_combination_x_y():
    x = torch.rand(size=(10, 10))
    y = torch.rand(size=(10, 10))
    xy = combination_x_y(x, y)
    print(xy.shape)


def test_combination_x_y_max():
    x = torch.rand(size=(BATCH_SIZE, 10, 10))
    y = torch.rand(size=(BATCH_SIZE, 10, 10))
    xy = combination_x_y_max(x, y)
    print(xy.shape)
