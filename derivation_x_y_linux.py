#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
在已知分配量a_equal_d的情况下，通过论文中买卖双方的报价x, y和分配量a_equal_d的关系，得到新一轮买卖双方的报价
2020/12/30
"""
from parameter_Iter_BAP_linux import *


def derivation_x_y(a_equal_d):
    x = torch.zeros(BATCH_SIZE, buyer_number, seller_number)
    y = torch.zeros(BATCH_SIZE, seller_number, buyer_number)
    x_fix_sum_part = torch.zeros(BATCH_SIZE, buyer_number, 1)
    y_fix_sum_part = torch.zeros(BATCH_SIZE, seller_number, 1)
    for batch_size_order in range(BATCH_SIZE):
        a_equal_d_convert = a_equal_d[batch_size_order].view(buyer_number, seller_number)

        for buyer_order in range(buyer_number):
            for seller_order in range(seller_number):
                x_fix_sum_part[batch_size_order][buyer_order][0] += (1 - z[buyer_order][seller_order])\
                                                                 / (a_equal_d_convert[buyer_order][seller_order]
                                                                    * (1 - z[buyer_order][seller_order]) + 1)
        for seller_order in range(seller_number):
            for buyer_order in range(buyer_number):
                y_fix_sum_part[batch_size_order][seller_order][0] += 2 * n_1[0][seller_order] * a_equal_d_convert[buyer_order][seller_order] + n_2[0][seller_order]

        for buyer_order in range(buyer_number):
            for seller_order in range(seller_number):
                x[batch_size_order][buyer_order][seller_order] = w[buyer_order][0] * a_equal_d_convert[buyer_order][seller_order] \
                                                                 * x_fix_sum_part[batch_size_order][buyer_order][0]
        for seller_order in range(seller_number):
            for buyer_order in range(buyer_number):
                y[batch_size_order][seller_order][buyer_order] = y_fix_sum_part[batch_size_order][seller_order][0] \
                                                                 / a_equal_d_convert[buyer_order][seller_order]
    return x, y


def test_derivation_x_y():
    a_equal_d = torch.rand(BATCH_SIZE, buyer_number * seller_number)
    x, y = derivation_x_y(a_equal_d=a_equal_d)
    print("推导后得到的x:", x)
    print("推导后得到的x的维度", x.shape)
    print("推导后得到的y:", y)
    print("推导后得到的y的维度", y.shape)




