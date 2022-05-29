#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
Iterative_BAP_nn_model里面的几个类别的全局参数
2020/01/02
"""
import torch
import numpy as np
import numpy.random as random
import time
from tensorboardX import SummaryWriter


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


set_seed(1)

"""
神经网络参数
"""
# D_in = 32 * 2 * 2
# H1 = 128
# H2 = 128
# D_out = 100
learning_rate = 1e-5
BATCH_SIZE = 2
decay_rate = 10
epoch_number = 100


"""
随机数据的参数
"""
data_set_creation_number = 1000
buyer_number = 10
seller_number = 10
proportion = 0.8
a_min = 5 + 5 * torch.rand(buyer_number, 1)
a_max = 12 + 6 * torch.rand(buyer_number, 1)
d_max = 10 + 10 * torch.rand(seller_number, 1)
index = random.randint(0, ((1 - proportion) * data_set_creation_number / BATCH_SIZE + 1))

"""
计算sw等的参数
"""
z = 0.01 * np.floor(random.uniform(1, 10, size=(buyer_number, seller_number)))       # z: distance
# constant = 25
# j_value = 1     # 这个值用来说明当crs变大的时候，即w变小的时候，社会福利会跟着变小
# crs = j_value + 3 + 5 * random.random(buyer_number)
# w = constant / crs
w_factor = 0.30
n_1_factor = 0.05
n_2_factor = 0.01
w = w_factor * np.floor(random.uniform(1, 10, size=(buyer_number, 1)))     # 等下试试这个w，这些都可以自己设置
n_1 = n_1_factor * np.floor(random.uniform(1, 5, size=(1, seller_number)))
n_2 = n_2_factor * np.floor(random.uniform(1, 5, size=(1, seller_number)))


"""
计算bm, sm的参数
"""
p_base_line = 7.45      # 简单一点，就可以直接确定一个基准价格
# base_line = np.random.randint(high=4, low=1, size=(buyer_number, 1))        # 复杂而且合理一点，就可以针对每一买家，都有不同的基准价格
r_base_line = 3.15
"""
可视化的参数
"""
begin_time = "{now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"), now_time=time.strftime("%H:%M:%S"))
# timestamp_win = "{now_day}_{now_time}_win_log".format(now_day=time.strftime("%Y_%m_%d"), now_time=time.strftime("%H_%M_%S"))
# SumWriter = SummaryWriter(log_dir="E:/experiment/Iter_BAP_Experiment_data/log_data/" + timestamp_win)        # windows log_dir
timestamp_linux = "{now_day}_{now_time}_linux_log".format(now_day=time.strftime("%Y_%m_%d"), now_time=time.strftime("%H_%M_%S"))
SumWriter = SummaryWriter(log_dir="./log_data/10_10/" + timestamp_linux)     # linux log_dir
