#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
import torch.nn as nn
import torch.nn.functional as f
import derivation_x_y_linux as derivation_x_y
import sw_Iter_BAP_linux as sw_iter_BAP
import bm_sm_Iter_BAP_linux as bm_sm_Iter_BAP
import combination_x_y_linux as combination_x_y
import codecs
import csv
import pickle
from parameter_Iter_BAP_linux import *

begin = time.clock()
with open('./objs_data/10_10/objs.pkl', 'rb') as objs_file:
    init_test_x_and_y, train_x_set, train_y_set, model_save_path = pickle.load(objs_file)
with open('./objs_data/10_10/objs_data_high_low_0201.pkl', 'rb') as objs_data_high_low_file:
    init_test_x_and_y_high_low = pickle.load(objs_data_high_low_file)
# with open('./objs_data/10_10/objs_data_y_high_low_01005.pkl', 'rb') as objs_data_high_low_file:
#     init_test_x_and_y_high_low = pickle.load(objs_data_high_low_file)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("mean of high_low:", init_test_x_and_y_high_low[0][0][1].mean())
# kk = input()


class NLayerNet(torch.nn.Module):
    def __init__(self):
        """
        :c1{Conv2d. Maxpool2d}: 卷积层，池化层
        :c2{Conv2d. Maxpool2d}: 卷积层，池化层
        :linear1: 线性层
        :linear2: 线性层
        :linear3: 线性层
        """
        super(NLayerNet, self).__init__()       # 定义卷积层，尝试加入卷积层（提取特征），池化层（降维），过于简单的神经网络，还有就是要尝试一下可视化,多去找找这方面的信息（书籍，视频）看看就是一个函数作为loss_function怎样避免loss丢失
        self.c1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
            ),      # 卷积后： (2 * 10 * 10) to (16 * 8 * 8)
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),      # 池化后： (16 * 8 * 8) to (16 * 4 * 4)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),      # 卷积后： (16 * 4 * 4) to (32 * 4 * 4)
            nn.LeakyReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),      # 池化后： (32 * 4 * 4) to (32 * 2 * 2)
        )
        self.linear1 = torch.nn.Linear(32 * 2 * 2, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 100)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x_y_price):       # 之前的想法都是在loss_function中加惩罚项约束，作用现在还不知道，但是现在的想法也可以在前向传播里加约束
        x_y_price = self.c1(x_y_price)
        x_y_price = self.dropout(x_y_price)
        x_y_price = self.c2(x_y_price)
        x_y_price = self.dropout(x_y_price)
        x_y_price = x_y_price.view(x_y_price.size(0), -1)
        x_y_price = self.linear1(x_y_price)
        x_y_price = f.leaky_relu(x_y_price)
        x_y_price = self.dropout(x_y_price)
        x_y_price = self.linear2(x_y_price)
        x_y_price = f.leaky_relu(x_y_price)
        x_y_price = self.dropout(x_y_price)
        x_y_price = self.linear3(x_y_price)
        x_y_price = 10 * f.sigmoid(x_y_price)
        return x_y_price


model_SGD = NLayerNet()
model_Momentum = NLayerNet()
model_RMSprop = NLayerNet()
model_Adam = NLayerNet()

checkpoint = torch.load(model_save_path)
model_SGD.load_state_dict(checkpoint['model_SGD_state_dict'])
model_Momentum.load_state_dict(checkpoint['model_Momentum_state_dict'])
model_RMSprop.load_state_dict(checkpoint['model_RMSprop_state_dict'])
model_Adam.load_state_dict(checkpoint['model_Adam_state_dict'])

model_SGD.to(device)
model_Momentum.to(device)
model_RMSprop.to(device)
model_Adam.to(device)

e = 0.0001
flag = 1
iteration = 100

sw_SGD_list = []
sw_sum_SGD_list = []
a_equal_d_SGD_list = []
u_SGD_list = []
p_SGD_list = []
r_SGD_list = []
c_SGD_list = []
bm_sum_SGD_list = []
sm_sum_SGD_list = []
pr_gap_sum_SGD_list = []

sw_Momentum_list = []
sw_sum_Momentum_list = []
a_equal_d_Momentum_list = []
u_Momentum_list = []
p_Momentum_list = []
r_Momentum_list = []
c_Momentum_list = []
bm_sum_Momentum_list = []
sm_sum_Momentum_list = []
pr_gap_sum_Momentum_list = []

sw_RMSprop_list = []
sw_sum_RMSprop_list = []
a_equal_d_RMSprop_list = []
u_RMSprop_list = []
p_RMSprop_list = []
r_RMSprop_list = []
c_RMSprop_list = []
bm_sum_RMSprop_list = []
sm_sum_RMSprop_list = []
pr_gap_sum_RMSprop_list = []

sw_Adam_list = []
sw_sum_Adam_list = []
a_equal_d_Adam_list = []
u_Adam_list = []
p_Adam_list = []
r_Adam_list = []
c_Adam_list = []
bm_sum_Adam_list = []
sm_sum_Adam_list = []
pr_gap_sum_Adam_list = []

test_x_SGD_aver_list = []
test_x_Momentum_aver_list = []
test_x_RMSprop_aver_list = []
test_x_Adam_aver_list = []
test_y_SGD_aver_list = []
test_y_Momentum_aver_list = []
test_y_RMSprop_aver_list = []
test_y_Adam_aver_list = []

# test_x_SGD_high_low_aver_list = []
# test_x_Momentum_high_low_aver_list = []
# test_x_RMSprop_high_low_aver_list = []
# test_x_Adam_high_low_aver_list = []
# test_y_SGD_high_low_aver_list = []
# test_y_Momentum_high_low_aver_list = []
# test_y_RMSprop_high_low_aver_list = []
# test_y_Adam_high_low_aver_list = []
"""
model_SGD迭代
"""

print("model_SGD开始作为数据处理器迭代，开始时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                 now_time=time.strftime("%H:%M:%S")))
for iteration_order in range(iteration):
    # 可以单个模型测试，四个模型测试有点不好弄，可以先弄一下
    if iteration_order == 0:
        # test_x_and_y = init_test_x_and_y
        test_x_and_y = init_test_x_and_y_high_low[0]
    test_x_and_y = test_x_and_y.to(device)
    model_SGD.eval()
    a_equal_d_SGD = model_SGD(test_x_and_y)  # test_x， a_equal_d， test_y这些对应的公式还没弄上去，还有数据的维度，类型还没弄好，先搭好一个框架
    a_equal_d_SGD = a_equal_d_SGD.cpu()
    print("a_equal_d_SGD:", a_equal_d_SGD)
    a_equal_d_SGD_aver = a_equal_d_SGD.sum() / (BATCH_SIZE * buyer_number * seller_number)
    SumWriter.add_scalar("test_a_equal_d_SGD_aver", a_equal_d_SGD_aver.item(),
                         global_step=iteration_order)
    sw_SGD, sw_sum_SGD = sw_iter_BAP.sw_iter_BAP(a_equal_d=a_equal_d_SGD, label="SGD", iteration=iteration_order)
    sw_SGD_list.append(sw_SGD)
    sw_sum_SGD_list.append(sw_sum_SGD.item())
    test_x, test_y = derivation_x_y.derivation_x_y(a_equal_d=a_equal_d_SGD)
    test_x_SGD_aver = test_x.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_x_SGD_aver_list.append(test_x_SGD_aver.item())        # 记录恶意报价x每次迭代的变化
    SumWriter.add_scalar("test_x_SGD_aver", test_x_SGD_aver.item(),
                         global_step=iteration_order)
    test_y_SGD_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_y_SGD_aver_list.append(test_y_SGD_aver.item())        # 记录恶意报价y每次迭代的变化
    SumWriter.add_scalar("test_y_SGD_aver", test_y_SGD_aver.item(),
                         global_step=iteration_order)
    bm_sum_SGD, sm_sum_SGD, pr_gap_sum_SGD, p_SGD, r_SGD = bm_sm_Iter_BAP.bm_sm_Iter_BAP(a_equal_d=a_equal_d_SGD,
                                                                                         input_x=test_x,
                                                                                         input_y=test_y,
                                                                                         label="SGD",
                                                                                         iteration=iteration_order)
    bm_sum_SGD_list.append(bm_sum_SGD.item())
    sm_sum_SGD_list.append(sm_sum_SGD.item())
    pr_gap_sum_SGD_list.append(pr_gap_sum_SGD.item())
    p_SGD_list.append(p_SGD.item())
    r_SGD_list.append(r_SGD.item())
    test_x_and_y = combination_x_y.combination_x_y_max(test_x, test_y)
    a_equal_d_SGD_list.append(a_equal_d_SGD.data.numpy())
    if iteration_order >= 1:
        # print("test_x_SGD:", test_x)
        # print("test_y_SGD:", test_y)
        if abs(sw_sum_SGD_list[iteration_order] - sw_sum_SGD_list[iteration_order - 1]) <= e:     # 收敛条件，sw收敛
            break
print("sw_sum_SGD:", sw_sum_SGD_list)
print("model_SGD成功作为数据处理器完成迭代，结束时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                   now_time=time.strftime("%H:%M:%S")))


"""
model_Momentum迭代
"""

print("model_Momentum开始作为数据处理器迭代，开始时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                      now_time=time.strftime("%H:%M:%S")))
for iteration_order in range(iteration):
    # 可以单个模型测试，四个模型测试有点不好弄，可以先弄一下
    if iteration_order == 0:
        # test_x_and_y = init_test_x_and_y
        test_x_and_y = init_test_x_and_y_high_low[0]
    test_x_and_y = test_x_and_y.to(device)
    model_Momentum.eval()
    a_equal_d_Momentum = model_Momentum(test_x_and_y)  # test_x， a_equal_d， test_y这些对应的公式还没弄上去，还有数据的维度，类型还没弄好，先搭好一个框架
    a_equal_d_Momentum = a_equal_d_Momentum.cpu()
    print("a_equal_d_Momentum:", a_equal_d_Momentum)
    a_equal_d_Momentum_aver = a_equal_d_Momentum.sum() / (BATCH_SIZE * buyer_number * seller_number)
    SumWriter.add_scalar("test_a_equal_d_Momentum_aver", a_equal_d_Momentum_aver.item(),
                         global_step=iteration_order)
    sw_Momentum, sw_sum_Momentum = sw_iter_BAP.sw_iter_BAP(a_equal_d=a_equal_d_Momentum, label="Momentum", iteration=iteration_order)
    sw_Momentum_list.append(sw_Momentum)
    sw_sum_Momentum_list.append(sw_sum_Momentum.item())
    test_x, test_y = derivation_x_y.derivation_x_y(a_equal_d=a_equal_d_Momentum)
    test_x_Momentum_aver = test_x.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_x_Momentum_aver_list.append(test_x_Momentum_aver.item())        # 记录恶意报价x每次迭代的变化
    SumWriter.add_scalar("test_x_Momentum_aver", test_x_Momentum_aver.item(),
                         global_step=iteration_order)
    test_y_Momentum_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_y_Momentum_aver_list.append(test_y_Momentum_aver.item())        # 记录恶意报价y每次迭代的变化
    SumWriter.add_scalar("test_y_Momentum_aver", test_y_Momentum_aver.item(),
                         global_step=iteration_order)
    bm_sum_Momentum, sm_sum_Momentum, pr_gap_sum_Momentum, p_Momentum, r_Momentum = bm_sm_Iter_BAP.bm_sm_Iter_BAP(a_equal_d=a_equal_d_Momentum,
                                                                                                                  input_x=test_x,
                                                                                                                  input_y=test_y,
                                                                                                                  label="Momentum",
                                                                                                                  iteration=iteration_order)
    bm_sum_Momentum_list.append(bm_sum_Momentum.item())
    sm_sum_Momentum_list.append(sm_sum_Momentum.item())
    pr_gap_sum_Momentum_list.append(pr_gap_sum_Momentum.item())
    p_Momentum_list.append(p_Momentum.item())
    r_Momentum_list.append(r_Momentum.item())
    test_x_and_y = combination_x_y.combination_x_y_max(test_x, test_y)
    a_equal_d_Momentum_list.append(a_equal_d_Momentum.data.numpy())
    if iteration_order >= 1:
        # print("test_x_Momentum:", test_x)
        # print("test_y_Momentum:", test_y)
        if abs(sw_sum_Momentum_list[iteration_order] - sw_sum_Momentum_list[iteration_order - 1]) <= e:     # 收敛条件，sw收敛
            break
print("sw_sum_Momentum:", sw_sum_Momentum_list)
print("model_Momentum成功作为数据处理器完成迭代，结束时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                        now_time=time.strftime("%H:%M:%S")))

"""
model_RMSprop迭代
"""

print("model_RMSprop开始作为数据处理器迭代，开始时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                     now_time=time.strftime("%H:%M:%S")))
for iteration_order in range(iteration):
    # 可以单个模型测试，四个模型测试有点不好弄，可以先弄一下
    if iteration_order == 0:
        # test_x_and_y = init_test_x_and_y
        test_x_and_y = init_test_x_and_y_high_low[0]
    test_x_and_y = test_x_and_y.to(device)
    model_RMSprop.eval()
    a_equal_d_RMSprop = model_RMSprop(test_x_and_y)  # test_x， a_equal_d， test_y这些对应的公式还没弄上去，还有数据的维度，类型还没弄好，先搭好一个框架
    a_equal_d_RMSprop = a_equal_d_RMSprop.cpu()
    print("a_equal_d_RMSprop:", a_equal_d_RMSprop)
    a_equal_d_RMSprop_aver = a_equal_d_RMSprop.sum() / (BATCH_SIZE * buyer_number * seller_number)
    SumWriter.add_scalar("test_a_equal_d_RMSprop_aver", a_equal_d_RMSprop_aver.item(),
                         global_step=iteration_order)
    sw_RMSprop, sw_sum_RMSprop = sw_iter_BAP.sw_iter_BAP(a_equal_d=a_equal_d_RMSprop, label="RMSprop", iteration=iteration_order)
    sw_RMSprop_list.append(sw_RMSprop)
    sw_sum_RMSprop_list.append(sw_sum_RMSprop.item())
    test_x, test_y = derivation_x_y.derivation_x_y(a_equal_d=a_equal_d_RMSprop)
    test_x_RMSprop_aver = test_x.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_x_RMSprop_aver_list.append(test_x_RMSprop_aver.item())        # 记录恶意报价x每次迭代的变化
    SumWriter.add_scalar("test_x_RMSprop_aver", test_x_RMSprop_aver.item(),
                         global_step=iteration_order)
    test_y_RMSprop_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_y_RMSprop_aver_list.append(test_y_RMSprop_aver.item())        # 记录恶意报价y每次迭代的变化
    SumWriter.add_scalar("test_y_RMSprop_aver", test_y_RMSprop_aver.item(),
                         global_step=iteration_order)
    bm_sum_RMSprop, sm_sum_RMSprop, pr_gap_sum_RMSprop, p_RMSprop, r_RMSprop = bm_sm_Iter_BAP.bm_sm_Iter_BAP(a_equal_d=a_equal_d_RMSprop,
                                                                                                             input_x=test_x,
                                                                                                             input_y=test_y,
                                                                                                             label="RMSprop",
                                                                                                             iteration=iteration_order)
    bm_sum_RMSprop_list.append(bm_sum_RMSprop.item())
    sm_sum_RMSprop_list.append(sm_sum_RMSprop.item())
    pr_gap_sum_RMSprop_list.append(pr_gap_sum_RMSprop.item())
    p_RMSprop_list.append(p_RMSprop.item())
    r_RMSprop_list.append(r_RMSprop.item())
    test_x_and_y = combination_x_y.combination_x_y_max(test_x, test_y)
    a_equal_d_RMSprop_list.append(a_equal_d_RMSprop.data.numpy())
    if iteration_order >= 1:
        # print("test_x_RMSprop:", test_x)
        # print("test_y_RMSprop:", test_y)
        if abs(sw_sum_RMSprop_list[iteration_order] - sw_sum_RMSprop_list[iteration_order - 1]) <= e:     # 收敛条件，sw收敛
            break
print("sw_sum_RMSprop:", sw_sum_RMSprop_list)
print("model_RMSprop成功作为数据处理器完成迭代，结束时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                       now_time=time.strftime("%H:%M:%S")))

"""
model_Adam迭代
"""

print("model_Adam开始作为数据处理器迭代，开始时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                  now_time=time.strftime("%H:%M:%S")))
for iteration_order in range(iteration):
    # 可以单个模型测试，四个模型测试有点不好弄，可以先弄一下
    if iteration_order == 0:
        # test_x_and_y = init_test_x_and_y
        test_x_and_y = init_test_x_and_y_high_low[0]
        # print("shape:", init_test_x_and_y_high_low[0].shape)
        # test_x_and_y = torch.ones(2, 2, 10, 10)
        # print("shape:", test_x_and_y.shape)
    test_x_and_y = test_x_and_y.to(device)
    model_Adam.eval()
    a_equal_d_Adam = model_Adam(test_x_and_y)  # test_x， a_equal_d， test_y这些对应的公式还没弄上去，还有数据的维度，类型还没弄好，先搭好一个框架
    a_equal_d_Adam = a_equal_d_Adam.cpu()
    print("a_equal_d_Adam:", a_equal_d_Adam)
    a_equal_d_Adam_aver = a_equal_d_Adam.sum() / (BATCH_SIZE * buyer_number * seller_number)
    SumWriter.add_scalar("test_a_equal_d_Adam_aver", a_equal_d_Adam_aver.item(),
                         global_step=iteration_order)
    sw_Adam, sw_sum_Adam = sw_iter_BAP.sw_iter_BAP(a_equal_d=a_equal_d_Adam, label="Adam", iteration=iteration_order)
    sw_Adam_list.append(sw_Adam)
    sw_sum_Adam_list.append(sw_sum_Adam.item())
    test_x, test_y = derivation_x_y.derivation_x_y(a_equal_d=a_equal_d_Adam)
    test_x_Adam_aver = test_x.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_x_Adam_aver_list.append(test_x_Adam_aver.item())        # 记录恶意报价x每次迭代的变化
    SumWriter.add_scalar("test_x_Adam_aver", test_x_Adam_aver.item(),
                         global_step=iteration_order)
    test_y_Adam_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
    test_y_Adam_aver_list.append(test_y_Adam_aver.item())        # 记录恶意报价y每次迭代的变化
    SumWriter.add_scalar("test_y_Adam_aver", test_y_Adam_aver.item(),
                         global_step=iteration_order)
    bm_sum_Adam, sm_sum_Adam, pr_gap_sum_Adam, p_Adam, r_Adam = bm_sm_Iter_BAP.bm_sm_Iter_BAP(a_equal_d=a_equal_d_Adam,
                                                                                              input_x=test_x,
                                                                                              input_y=test_y,
                                                                                              label="Adam",
                                                                                              iteration=iteration_order)
    bm_sum_Adam_list.append(bm_sum_Adam.item())
    sm_sum_Adam_list.append(sm_sum_Adam.item())
    pr_gap_sum_Adam_list.append(pr_gap_sum_Adam.item())
    p_Adam_list.append(p_Adam.item())
    r_Adam_list.append(r_Adam.item())
    test_x_and_y = combination_x_y.combination_x_y_max(test_x, test_y)
    a_equal_d_Adam_list.append(a_equal_d_Adam.data.numpy())
    if iteration_order >= 1:
        # print("test_x_Adam:", test_x)
        # print("test_y_Adam:", test_y)
        if abs(sw_sum_Adam_list[iteration_order] - sw_sum_Adam_list[iteration_order - 1]) <= e:     # 收敛条件，sw收敛
            break
print("sw_sum_Adam:", sw_sum_Adam_list)
print("model_Adam成功作为数据处理器完成迭代，结束时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                    now_time=time.strftime("%H:%M:%S")))


"""
将list数据保存至csv, txt文件中
"""


def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("国家级数据保存在{file_name}文件中，保存文件成功".format(file_name=file_name))


"""
保存文件的位置(linux版本)
"""
data_write_csv("./list_data/10_10/Iter_a_SGD.csv", a_equal_d_SGD)
data_write_csv("./list_data/10_10/Iter_Momentum_a.csv", a_equal_d_Momentum)
data_write_csv("./list_data/10_10/Iter_RMSprop_a.csv", a_equal_d_RMSprop)
data_write_csv("./list_data/10_10/Iter_Adam_a.csv", a_equal_d_Adam)


txt_name_list = ["./list_data/10_10/Iter_sw_SGD.txt",
                 "./list_data/10_10/Iter_sw_sum_SGD.txt",
                 "./list_data/10_10/Iter_bm_sum_SGD.txt",
                 "./list_data/10_10/Iter_sm_sum_SGD.txt",
                 "./list_data/10_10/Iter_pr_gap_sum_SGD.txt",
                 "./list_data/10_10/Iter_u_SGD.txt",
                 "./list_data/10_10/Iter_p_SGD.txt",
                 "./list_data/10_10/Iter_r_SGD.txt",
                 "./list_data/10_10/Iter_c_SGD.txt",     # SGD_list
                 "./list_data/10_10/Iter_sw_Momentum.txt",
                 "./list_data/10_10/Iter_sw_sum_Momentum.txt",
                 "./list_data/10_10/Iter_bm_sum_Momentum.txt",
                 "./list_data/10_10/Iter_sm_sum_Momentum.txt",
                 "./list_data/10_10/Iter_pr_gap_sum_Momentum.txt",
                 "./list_data/10_10/Iter_u_Momentum.txt",
                 "./list_data/10_10/Iter_p_Momentum.txt",
                 "./list_data/10_10/Iter_r_Momentum.txt",
                 "./list_data/10_10/Iter_c_Momentum.txt",        # Momentum_list
                 "./list_data/10_10/Iter_sw_RMSprop.txt",
                 "./list_data/10_10/Iter_sw_sum_RMSprop.txt",
                 "./list_data/10_10/Iter_bm_sum_RMSprop.txt",
                 "./list_data/10_10/Iter_sm_sum_RMSprop.txt",
                 "./list_data/10_10/Iter_pr_gap_sum_RMSprop.txt",
                 "./list_data/10_10/Iter_u_RMSprop.txt",
                 "./list_data/10_10/Iter_p_RMSprop.txt",
                 "./list_data/10_10/Iter_r_RMSprop.txt",
                 "./list_data/10_10/Iter_c_RMSprop.txt",     # RMSprop_list
                 "./list_data/10_10/Iter_sw_Adam.txt",
                 "./list_data/10_10/Iter_sw_sum_Adam.txt",
                 "./list_data/10_10/Iter_bm_sum_Adam.txt",
                 "./list_data/10_10/Iter_sm_sum_Adam.txt",
                 "./list_data/10_10/Iter_pr_gap_sum_Adam.txt",
                 "./list_data/10_10/Iter_u_Adam.txt",
                 "./list_data/10_10/Iter_p_Adam.txt",
                 "./list_data/10_10/Iter_r_Adam.txt",
                 "./list_data/10_10/Iter_c_Adam.txt",         # Adam_list
                 "./list_data/10_10/Iter_x_SGD.txt",
                 "./list_data/10_10/Iter_x_Momentum.txt",
                 "./list_data/10_10/Iter_x_RMSprop.txt",
                 "./list_data/10_10/Iter_x_Adam.txt",
                 "./list_data/10_10/Iter_y_SGD.txt",
                 "./list_data/10_10/Iter_y_Momentum.txt",
                 "./list_data/10_10/Iter_y_RMSprop.txt",
                 "./list_data/10_10/Iter_y_Adam.txt"]          # 恶意报价x_list,y_list

txt_list = [sw_SGD_list, sw_sum_SGD_list, bm_sum_SGD_list, sm_sum_SGD_list, pr_gap_sum_SGD_list, u_SGD_list, p_SGD_list,
            r_SGD_list, c_SGD_list, sw_Momentum_list, sw_sum_Momentum_list,  bm_sum_Momentum_list, sm_sum_Momentum_list,
            pr_gap_sum_Momentum_list, u_Momentum_list, p_Momentum_list, r_Momentum_list, c_Momentum_list,
            sw_RMSprop_list, sw_sum_RMSprop_list, bm_sum_RMSprop_list, sm_sum_RMSprop_list, pr_gap_sum_RMSprop_list,
            u_RMSprop_list, p_RMSprop_list, r_RMSprop_list, c_RMSprop_list, sw_Adam_list, sw_sum_Adam_list,
            bm_sum_Adam_list, sm_sum_Adam_list, pr_gap_sum_Adam_list, u_Adam_list, p_Adam_list, r_Adam_list,
            c_Adam_list, test_x_SGD_aver_list, test_x_Momentum_aver_list, test_x_Momentum_aver_list,
            test_x_Adam_aver_list, test_y_SGD_aver_list, test_y_Momentum_aver_list, test_y_Momentum_aver_list,
            test_y_Adam_aver_list]

for txt_number in range(44):                                    # 原先的36 + 后来的8 = 44
    filename = open(txt_name_list[txt_number], "w")
    # sw_order = 0
    for value in txt_list[txt_number]:
        # filename.write("数据{sw_order}:".format(sw_order=sw_order + 1))
        # filename.write("  ")
        filename.write(str(value))
        filename.write("\n")
        # sw_order += 1
    filename.close()


"""
打印出p,r的数据，知道p,r的大小而不是只知道p-r
"""

print("p_SGD_list: ")
print(p_SGD_list)
print("r_SGD_list: ")
print(r_SGD_list)
print("p_Momentum_list: ")
print(p_Momentum_list)
print("r_Momentum_list: ")
print(r_Momentum_list)
print("p_RMSprop_list: ")
print(p_RMSprop_list)
print("r_RMSprop_list: ")
print(r_RMSprop_list)
print("p_Adam_list: ")
print(p_Adam_list)
print("r_Adam_list: ")
print(r_Adam_list)
print("test_y_Adam_aver:")
print(test_y_Adam_aver_list)


print("epoch：{epoch}    BATCH_SIZE: {BATCH_SIZE}    b_n: {buyer_number}    p_b_l: {p_base_line}    "
      "r_b_l: {r_base_line}     w: {w}      m: {m}      n: {n}"
      .format(epoch=epoch_number,
              BATCH_SIZE=BATCH_SIZE,
              buyer_number=buyer_number,
              p_base_line=p_base_line,
              r_base_line=r_base_line,
              w=w_factor,
              m=n_1_factor,
              n=n_2_factor))       # 记录重要参数，方便在运行多个实验的时候知道这个实验是哪个实验
print("x_mean: {one_train_x_mean}      y_mean: {one_train_y_mean} "
      "x_std: {one_train_x_std}     y_std: {one_train_y_std}"
      .format(one_train_x_mean=train_x_set[200].mean(),
              one_train_y_mean=train_y_set[200].mean(),
              one_train_x_std=train_x_set[200].std(),
              one_train_y_std=train_y_set[200].std()))

end = time.clock()      # 程序结束计时
print('双拍算法时间： {test_time}s'.format(test_time=(end - begin)))
print('程序开始的时刻： {begin_time}'.format(begin_time=begin_time))
print('程序结束的时刻： {now_day}\t{now_time}'.format(now_day=time.strftime("%Y/%m/%d"), now_time=time.strftime("%H:%M:%S")))