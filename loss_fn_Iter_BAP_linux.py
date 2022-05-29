#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
input_x,input_y都是一个维度为(BATCH_SIZE, buyer_number, seller_number), input_a_d是一个维度为（BATCH_SIZE, buyer_number * seller_number)
已知这三个变量，以BAP公式作为框架设计该神经网络的loss_function
以后要养成习惯，像这个程序一样，在设计完辅助程序的时候，在后面也罢测试的代码整理成规范的test程序
2020/01/02
"""
from parameter_Iter_BAP_linux import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def loss_fn_Iter_BAP(input_x, input_y, input_a_d):
    bap_value_list = []
    bap_value_sum = torch.zeros(1).to(device)
    input_y_tran = input_y.transpose(1, 2)
    for batch_size_order in range(BATCH_SIZE):
        input_a_d_convert = input_a_d[batch_size_order].view(buyer_number, seller_number)
        bap_value_temp = -(input_x[batch_size_order] * torch.log(input_a_d_convert)
                           - 1/2 * input_y_tran[batch_size_order] * (input_a_d_convert ** 2))
        bap_value = bap_value_temp.sum()
        bap_value_list.append(bap_value)
        bap_value_sum += bap_value
    return bap_value_sum


def test_loss_fn_Iter_BAP():
    x = torch.rand((BATCH_SIZE, buyer_number, seller_number))

    y = torch.rand((BATCH_SIZE, buyer_number, seller_number))
    a_d = torch.rand((BATCH_SIZE, 100))
    bap_value_sum = loss_fn_Iter_BAP(input_x=x, input_y=y, input_a_d=a_d)
    print(x)
    print("BAP_value_sum:", bap_value_sum)
    print("BAP_value_sum的维度:", bap_value_sum.shape)


