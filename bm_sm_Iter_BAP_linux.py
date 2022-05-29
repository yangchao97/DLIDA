"""
在已知a_equal_d, x, y的情况下， 计算并且记录四个量，买家的满意度U, 买家的支付规则P，以及卖家的成本C，卖家的奖励规则R，并且通过公式(U-C)
解决了买家的BMP，同样的通过公式(R-C)解决了卖家的SMP
2020/01/11
"""
from parameter_Iter_BAP_linux import *


def bm_sm_Iter_BAP(a_equal_d, input_x, input_y, label, iteration):
    input_y_tran = input_y.transpose(1, 2)
    w_convert = torch.from_numpy(w)
    z_convert = torch.from_numpy(z)
    w_convert = w_convert.clone().detach().type(dtype=torch.float32)
    z_convert = z_convert.clone().detach().type(dtype=torch.float32)
    n_1_convert = torch.from_numpy(n_1)
    n_2_convert = torch.from_numpy(n_2)
    n_1_convert = n_1_convert.clone().detach().type(dtype=torch.float32)
    n_2_convert = n_2_convert.clone().detach().type(dtype=torch.float32)
    bm_sum = 0
    sm_sum = 0
    pr_gap_sum = 0
    p_sum = 0
    r_sum = 0
    # u = []
    # p = []
    # r = []
    # c = []
    for batch_size_order in range(BATCH_SIZE):
        a_equal_d_convert = a_equal_d[batch_size_order].view(buyer_number, seller_number)
        u_temp = w_convert * torch.log(a_equal_d_convert - a_equal_d_convert * z_convert + 1)
        p_temp = input_x[batch_size_order] - p_base_line
        r_temp = a_equal_d_convert ** 2 * input_y_tran[batch_size_order] - r_base_line
        c_temp = n_1_convert * (a_equal_d_convert ** 2) + n_2_convert * a_equal_d_convert
        u_single = u_temp.sum()
        p_single = p_temp.sum()
        r_single = r_temp.sum()
        c_single = c_temp.sum()
        bm_temp = u_single - p_single
        sm_temp = r_single - c_single
        pr_gap_temp = p_single - r_single
        bm_sum += bm_temp
        sm_sum += sm_temp
        pr_gap_sum += pr_gap_temp
        p_sum += p_single
        r_sum += r_single
        # u.append(u_single)
        # p.append(p_single)
        # r.append(r_single)
        # c.append(c_single)
    bm_sum = bm_sum / BATCH_SIZE
    sm_sum = sm_sum / BATCH_SIZE
    pr_gap_sum = pr_gap_sum / BATCH_SIZE
    p_sum = p_sum / BATCH_SIZE
    r_sum = r_sum / BATCH_SIZE
    SumWriter.add_scalar("BM_PLUS_" + label, bm_sum, global_step=iteration)
    SumWriter.add_scalar("SM_PLUS_" + label, sm_sum, global_step=iteration)
    SumWriter.add_scalar("PR_GAP_PLUS_" + label, pr_gap_sum, global_step=iteration)
    return bm_sum, sm_sum, pr_gap_sum, p_sum, r_sum


def test_bm_sm_Iter_BAP():
    a_equal_d = torch.rand(size=(BATCH_SIZE, buyer_number * seller_number))
    input_x = 7 * torch.rand(size=(BATCH_SIZE, buyer_number, seller_number))
    input_y = 2 * torch.rand(size=(BATCH_SIZE, seller_number, buyer_number))
    bm_sum, sm_sum, pr_gap_sum, p, r = bm_sm_Iter_BAP(a_equal_d=a_equal_d, input_x=input_x, input_y=input_y, label="SGD", iteration=2)
    print("\n")
    print("bm_sum: ", bm_sum)
    print("sm_sum: ", sm_sum)
    print("pr_gap_sum: ", pr_gap_sum)



