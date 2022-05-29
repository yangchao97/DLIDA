"""
在已知a_equal_d的前提下，计算sw,同时可能也需要计算SW，同时也要为之后保存sw，SW铺垫
2020/01/04
"""
from parameter_Iter_BAP_linux import *


def sw_iter_BAP(a_equal_d, label, iteration):
    w_convert = torch.from_numpy(w)
    z_convert = torch.from_numpy(z)
    w_convert = w_convert.clone().detach().type(dtype=torch.float32)
    z_convert = z_convert.clone().detach().type(dtype=torch.float32)
    n_1_convert = torch.from_numpy(n_1)
    n_2_convert = torch.from_numpy(n_2)
    n_1_convert = n_1_convert.clone().detach().type(dtype=torch.float32)
    n_2_convert = n_2_convert.clone().detach().type(dtype=torch.float32)
    sw = []     # (buyer_number * seller_number)个数据的社会福利
    sw_sum = 0     # BATCH_SIZE个sw的和
    for batch_size_order in range(BATCH_SIZE):
        a_equal_d_convert = a_equal_d[batch_size_order].view(buyer_number, seller_number)
        # print("w_convert的type:", w_convert.type())
        # print("z_convert的type:", z_convert.type())
        # print("a_convert的type:", a_equal_d_convert.type())
        sw_temp = (w_convert * torch.log(a_equal_d_convert - a_equal_d_convert * z_convert + 1)
                   - (n_1_convert * (a_equal_d_convert ** 2) + n_2_convert * a_equal_d_convert)).sum()
        # SumWriter.add_scalar("sw_" + label, sw_temp, global_step=iteration * BATCH_SIZE + batch_size_order)
        sw.append(sw_temp)
        sw_sum += sw_temp
    sw_sum = sw_sum / BATCH_SIZE
    SumWriter.add_scalar("SW_PLUS_" + label, sw_sum, global_step=iteration)
    return sw, sw_sum


def test_sw_iter_BAP():
    a_equal_d = torch.rand([BATCH_SIZE, buyer_number * seller_number])
    sw_sgd, sw_sum_sgd = sw_iter_BAP(a_equal_d, "SGD", 3)
    print("\n")
    print("a_equal_d的维度：", a_equal_d.shape)
    print("sw_sgd:", sw_sgd)
    print("sw_sgd的维度：", len(sw_sgd))
    print("sw_sum_sgd:", sw_sum_sgd)
    print("sw_sum_sgd的维度：", sw_sum_sgd.shape)
