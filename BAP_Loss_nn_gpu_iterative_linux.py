#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
本code相对于BAP_Loss_nn_model理论更加合理，结构更加优化，BAP_Loss_nn_model只采用一次神经网络，输入的数据是matlab提取的最后一层的
买卖双方的报价，因此造成的问题是 1：极度依赖于matlab中提取的数据，matlab的数据会影响模型的品质；2：因为只提取了最后一层的买卖双方的
报价，这使得神经网络只能处理一次数据，失去了双边拍卖框架迭代更新优化的特色，无法发挥双边拍卖的优势。
本code针对这两个问题，给出了解决方案 1：自定义输入的原始买卖双方的报价数据，只要报价满足随机性和约束条件，就是可以成立的，这也是这个
领域的特殊性，没有公共的数据集，可以自定义数据集；2：把神经网络作为一个数据处理器，这个数据处理器的作用是输入买卖双方的报价，能得到使
得BAP最大化的a_equal_d，然后再通过买卖双方的分配信息和报价之间的关系得到新一轮的报价，知道最后迭代收敛，这既保留了双边拍卖的特色，又
同时引入了神经网络。
目前本code还有两个需要优化的地方 1：数据或者网络结构通过tensorboard或者其他手段可视化，而不是通过list；2：神经网络中涉及的参数太多，
可能需要用Dropout的方法优化神经网络，防止过拟合。
2020/12/29
"""
import torch.nn as nn
import torch.nn.functional as f
import creation_data_x_y_linux as creation_data_x_y
import creation_data_x_y_other_linux as creation_data_x_y_other
import derivation_x_y_linux as derivation_x_y
import combination_x_y_linux as combination_x_y
import loss_fn_Iter_BAP_linux as loss_fn_Iter_BAP
import sw_Iter_BAP_linux as sw_iter_BAP
import bm_sm_Iter_BAP_linux as bm_sm_Iter_BAP
from parameter_Iter_BAP_linux import *
import codecs
import csv
from torch.utils.data import Dataset, DataLoader
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
begin = time.clock()        # 程序开始计时
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

"""
ready the dataset
"""

"""
generate the data and divide the data
数据可以自定义，不需要用matlab中的数据，只需要数据符合BAP的约束，而且matlab中得到的数据一直都是最后一次迭代中的报价信息x,y和分配信息a,d
另外就是它里面的数据对于训练神经网络可能太少，可以自定义随机数数据大小，随意的取足够的数据，毕竟大家都没有使用公开的数据集
"""

"""
通过data_x_y.data_x_y_creation得到数据集
"""
train_x_set = []
train_y_set = []
test_x_set = []
test_y_set = []
train_x_set, train_y_set, test_x_set, test_y_set = creation_data_x_y.creation_data_x_y()  # (正态分布数据集)买卖双方分别生成10000组报价数据，每组报价数据包含数据个数10*10，train_set：all_data_set = 0.8
# train_x_set, train_y_set, test_x_set, test_y_set = creation_data_x_y_other.creation_data_x_y_other()  # (随机分布数据集)买卖双方分别生成10000组报价数据，每组报价数据包含数据个数10*10，train_set：all_data_set = 0.8
""""
加载train_data, test_data
"""


class TrainData(Dataset):
    def __init__(self):
        all_train_data = []
        for (train_x_data, train_y_data) in zip(train_x_set, train_y_set):
            all_train_data.append((train_x_data, train_y_data))
        self.all_train_data = all_train_data

    def __getitem__(self, train_data_index):
        x_train, y_train = self.all_train_data[train_data_index]
        xy_train = combination_x_y.combination_x_y(x_train, y_train)
        return x_train, y_train, xy_train

    def __len__(self):
        return len(self.all_train_data)


"""
程序后面是随机提取一个10*10的数据，不需要dataset或者dataloader来提取数据，或许也需要，因为一直训练都是以batch的形式训练，可能模型更加适应相同BATCH_SIZE的数据
"""


class TestData(Dataset):
    def __init__(self):
        all_test_data = []
        for (test_x_data, test_y_data) in zip(test_x_set, test_y_set):
            all_test_data.append((test_x_data, test_y_data))
        self.all_test_data = all_test_data

    def __getitem__(self, test_data_index):
        x_test, y_test = self.all_test_data[test_data_index]
        xy_test = combination_x_y.combination_x_y(x_test, y_test)
        return x_test, y_test, xy_test

    def __len__(self):
        return len(self.all_test_data)


train_data = TrainData()
test_data = TestData()

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)       # 不打乱数据，其实也可以打乱数据，因为数据的打乱形式只是打乱10*10以外的顺序，10*10里面的顺序是固定的


"""
the neural network architecture
现在用的神经网络框架还是简单的三层全连接神经网络，要尝试采用其他神经网络，依次排序 LSTM(考虑时间顺序在迭代的时候text_x,text_y) > rnn > cnn，这种神经网络会出现loss丢失的情况
另外一个功能要实现的是就是在这个神经网络中要可能实验关于输出的a,d的约束的关系，可以在最后一层神经网络中进行约束，或者也可以尝试在loss_function中加入有关约束的惩罚项进行约束
还有一直都没有采用dropout这种技巧优化网络
还有通过tensorboard实现可视化
"""


"""
还有按照现在的神经网络框架，神经网络里面的参数太多了，可能确实需要考虑dropout方法随机失活神经元，使得神经网络的泛化能力更强，之前没考虑，
现在要去看看dropout相关的demo看看如何放置
"""


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
models = [model_SGD, model_Momentum, model_RMSprop, model_Adam]
model_SGD.to(device)
model_Momentum.to(device)
model_RMSprop.to(device)
model_Adam.to(device)

optimizer_SGD = torch.optim.SGD(model_SGD.parameters(), lr=learning_rate)
optimizer_Momentum = torch.optim.SGD(model_Momentum.parameters(), lr=learning_rate, momentum=0.9)
optimizer_RMSprop = torch.optim.RMSprop(model_RMSprop.parameters(), lr=learning_rate, alpha=0.9)
optimizer_Adam = torch.optim.Adam(model_Adam.parameters(), lr=learning_rate, betas=(0.9, 0.999))
optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSprop, optimizer_Adam]
opt_labels = ["SGD", "Momentum", "RMSprop", "Adam"]


"""
神经网络结构可视化, HiddenLayer(内置函数总是出错,要求torchvision的版本，因而也会要去torch的版本), PyTorchViz(也是出一些不知道的错，查不到怎么解决)，tensorboard
只能采用tensorboard或者Visdom
"""
"""
# 这段代码用来加载神经网络的graph，亲测能加载，但是要求tensorboard的版本在1.14及以上，但是1.14之上的版本也要和tensorflow这些工具包一起配套升级就很麻烦，
# 不过有一种比较狡猾的方法，就是可以先用1.14之上的版本先生成好log，然后再还原成1.10的版本(我的版本是1.10)加载log可视化，然后把graph下载，毕竟
# 网络结构不经常变，但是那些scalar则经常需要调整，所以就可以还原成1.10继续可视化scalar
graph_data = torch.rand(size=(1, 2, 10, 10), dtype=torch.float32).to(device)
SumWriter.add_graph(model_SGD, input_to_model=graph_data)
"""


"""
training the neural network
"""

for epoch in range(epoch_number):
    learning_rate = 1 / (1 + decay_rate * epoch) * learning_rate
    for i, get_data in enumerate(train_loader, 0):
        model_time = 0
        for (model, opt) in zip(models, optimizers):
            buyer_x, seller_y, x_and_y = get_data
            buyer_x = buyer_x.clone().detach().type(dtype=torch.float32).to(device)     # 维度应该是(BATCH_SIZE， buyer_number， seller_number)
            seller_y = seller_y.clone().detach().type(dtype=torch.float32).to(device)      # 维度应该是(BATCH_SIZE， buyer_number， seller_number)
            x_and_y = x_and_y.to(device)
            """
            a_equal_d = model(x_and_y)出问题了，大部分a_equal_d里面的元素都为0， 且后面的a_equal_d越来越多0
            """
            model.train()
            a_equal_d = model(x_and_y)      # a_equal_d的维度应该是(BATCH_SIZE, 100,)
            print("buyer_x:", buyer_x)
            print("seller_y:", seller_y)
            print("a_equal_d:", a_equal_d)
            a_equal_d_aver = a_equal_d.sum() / (BATCH_SIZE * buyer_number * seller_number)
            SumWriter.add_scalar(opt_labels[model_time] + " a_equal_d_aver", a_equal_d_aver.item(),
                                 global_step=epoch * len(train_loader) + i)
            loss = loss_fn_Iter_BAP.loss_fn_Iter_BAP(buyer_x, seller_y, a_equal_d)        # 这里的loss还没有设计好，维度还没有对应上去，应该要使用for循环，如果对tensor的计算不熟悉的话，如果熟悉基本都是使用点乘
            print("epoch: {epoch}    step: {step}   lr: {lr}".format(epoch=epoch, step=i, lr=learning_rate))
            print("优化方法为：{opt}    loss: {loss}     a_equal_d_aver；{a_equal_d_aver}"
                  .format(opt=opt_labels[model_time], loss=loss, a_equal_d_aver=a_equal_d_aver))
            print("\n")
            SumWriter.add_scalar(opt_labels[model_time] + " loss", loss.item(),
                                 global_step=epoch * len(train_loader) + i)
            model_time += 1
            opt.zero_grad()
            loss.backward()
            opt.step()
"""
将训练好的神经网络模型作为与BAP功能一样的数据处理器，然后取代BAP构建迭代双边拍卖框架直至收敛,中断循环的条件--当报价或者分配信息收敛的时候
这里的应该只有一组test_data，循环代入，所以之前的test_data, test_loader都可以不要，
这里一直考虑的都是输入模型中的是买家输入bn * sn的数据，卖家输入bn * sn的数据,如果batch ！= 1的话，输入数据则是 原本的数据 * batch_size，
这里可能就不会采取原来的输入维度为2， 会采取输入维度为200，这样也可能可以用卷积神经网络改善神经网络的构架
"""

"""
在这里实现tensorboard或者其他可视化，不要一直回避可视化
"""
start = time.clock()
for j, get_data in enumerate(test_loader, 0):
    init_test_x, init_test_y, init_test_x_and_y = get_data
    init_test_x = init_test_x.clone().detach().type(dtype=torch.float32).to(device)      # 之后的程序出了model是GPU之外，其他都是CPU，这里会不会冲突，可能需要把之后的程序都统一为CPU或者GPU
    init_test_y = init_test_y.clone().detach().type(dtype=torch.float32).to(device)
    init_test_x_and_y = init_test_x_and_y.to(device)
    if j == index:
        break

e = 0.0001
flag = 1
iteration = 100

sw_SGD_list = []        # u,c的list在后面的版本中都删去了，但是要改的信息太多了，且不影响实验结果，就暂时不删去
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
"""
model_SGD迭代
"""

print("model_SGD开始作为数据处理器迭代，开始时刻为： {now_day}\t{now_time}".format(now_day=time.strftime("%Y/%m/%d"),
                                                                 now_time=time.strftime("%H:%M:%S")))
for iteration_order in range(iteration):
    # 可以单个模型测试，四个模型测试有点不好弄，可以先弄一下
    if iteration_order == 0:
        test_x_and_y = init_test_x_and_y
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
    SumWriter.add_scalar("test_x_SGD_aver", test_x_SGD_aver.item(),
                         global_step=iteration_order)
    test_y_SGD_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
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
        test_x_and_y = init_test_x_and_y
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
    SumWriter.add_scalar("test_x_Momentum_aver", test_x_Momentum_aver.item(),
                         global_step=iteration_order)
    test_y_Momentum_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
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
        test_x_and_y = init_test_x_and_y
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
    SumWriter.add_scalar("test_x_RMSprop_aver", test_x_RMSprop_aver.item(),
                         global_step=iteration_order)
    test_y_RMSprop_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
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
        test_x_and_y = init_test_x_and_y
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
    SumWriter.add_scalar("test_x_Adam_aver", test_x_Adam_aver.item(),
                         global_step=iteration_order)
    test_y_Adam_aver = test_y.sum() / (BATCH_SIZE * buyer_number * seller_number)
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
                 "./list_data/10_10/Iter_c_Adam.txt"]        # Adam_list

txt_list = [sw_SGD_list, sw_sum_SGD_list, bm_sum_SGD_list, sm_sum_SGD_list, pr_gap_sum_SGD_list, u_SGD_list, p_SGD_list,
            r_SGD_list, c_SGD_list, sw_Momentum_list, sw_sum_Momentum_list,  bm_sum_Momentum_list, sm_sum_Momentum_list,
            pr_gap_sum_Momentum_list, u_Momentum_list, p_Momentum_list, r_Momentum_list, c_Momentum_list,
            sw_RMSprop_list, sw_sum_RMSprop_list, bm_sum_RMSprop_list, sm_sum_RMSprop_list, pr_gap_sum_RMSprop_list,
            u_RMSprop_list, p_RMSprop_list, r_RMSprop_list, c_RMSprop_list, sw_Adam_list, sw_sum_Adam_list,
            bm_sum_Adam_list, sm_sum_Adam_list, pr_gap_sum_Adam_list, u_Adam_list, p_Adam_list, r_Adam_list,
            c_Adam_list]
for txt_number in range(36):
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

print("本次实验的重要参数如下：")
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
print("x_mean: {one_train_x_mean}      y_mean: {one_train_y_mean}      "
      "x_std: {one_train_x_std}     y_std: {one_train_y_std}"
      .format(one_train_x_mean=train_x_set[200].mean(),
              one_train_y_mean=train_y_set[200].mean(),
              one_train_x_std=train_x_set[200].std(),
              one_train_y_std=train_y_set[200].std()))

end = time.clock()      # 程序结束计时
print('程序用时1（排除模型训练的时间）： {test_time}s'.format(test_time=(end - start)))
print('程序用时2： {all_time}s'.format(all_time=end - begin))
print('程序开始的时刻： {begin_time}'.format(begin_time=begin_time))
print('程序结束的时刻： {now_day}\t{now_time}'.format(now_day=time.strftime("%Y/%m/%d"), now_time=time.strftime("%H:%M:%S")))

