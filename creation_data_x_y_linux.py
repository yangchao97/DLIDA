#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
生成服从高斯分布（正态分布）的报价数据，
将符合条件的报价数据进行
2021/04/28
"""
import scipy.io as sio
from parameter_Iter_BAP_linux import *


def boxMullerSampling(mu, sigma, size):
    """
    利用Box-Muller算法生成正态分布随机数生成函数
    """
    u = np.random.uniform(size=size)
    v = np.random.uniform(size=size)
    z = np.sqrt(-2 * np.log(u)) * np.cos(2 * np.pi * v)
    return mu + z * sigma


def creation_data_x_y():
    mu_init_x = 2
    sigma_init_x = 1
    mu_init_y = 1
    sigma_init_y = 0.5
    size_init = 110000
    x_creation_data = boxMullerSampling(mu_init_x, sigma_init_x, size_init)
    y_creation_data = boxMullerSampling(mu_init_y, sigma_init_y, size_init)
    print("x_data的维度", x_creation_data.shape)
    print("x_data的均值和方差：", x_creation_data.mean(), x_creation_data.std())
    print("y_data的数量：", len(y_creation_data))
    print("y_data的维度", y_creation_data.shape)
    print("y_data的均值和方差：", y_creation_data.mean(), y_creation_data.std())
    print("y_data的数量：", len(y_creation_data))

    """
    采样生成一定数量的随机数并且保存
    方式1：首先出现的符合条件的数据直接保存，保存到一定数量(b_n*s_n)后停止采样;
    方式2：先保存全部符合条件的数据并且排序，然后根据符合条件的数量按照均匀采样的方式进行采样；
    采样好的数据都需要通过转化为张量，为了和后面的数据对接
    """
    x_set_order = 0
    y_set_order = 0
    x_data_set_order = 0
    y_data_set_order = 0
    train_x_set = []
    test_x_set = []
    train_y_set = []
    test_y_set = []
    for value_order in range(size_init):
        """
        x_set放在循环里且每次一定条件下都要重新创建，是因为需要每次append的地址都是一个新的地址，要不然append的数据会被最后一个数据覆盖
        """
        if x_set_order == 0:
            x_set = torch.zeros([buyer_number * seller_number])
            y_set = torch.zeros([buyer_number * seller_number])
        if x_data_set_order == data_set_creation_number and y_data_set_order == data_set_creation_number:
            break
        else:
            # if not((mu_init_x - 2 * sigma_init_x) <= x_creation_data[value_order] <= (mu_init_x + 2 * sigma_init_x) or
            #        (mu_init_y - 2 * sigma_init_y) <= y_creation_data[value_order] <= (mu_init_y + 2 * sigma_init_y)):
            #     continue
            if x_data_set_order != data_set_creation_number:
                if (mu_init_x - 2 * sigma_init_x) <= x_creation_data[value_order] <= (mu_init_x + 2 * sigma_init_x):
                    x_set[x_set_order] = x_creation_data[value_order]
                    x_set_order += 1
                if x_set_order == buyer_number * seller_number:
                    x_data_set_order += 1
                    if x_data_set_order <= (data_set_creation_number * proportion):
                        train_x_set.append(x_set.reshape(buyer_number, seller_number))
                    else:
                        test_x_set.append(x_set.reshape(buyer_number, seller_number))
                    x_set_order = 0
            if y_data_set_order != data_set_creation_number:
                if (mu_init_y - 2 * sigma_init_y) <= y_creation_data[value_order] <= (mu_init_y + 2 * sigma_init_y):
                    y_set[y_set_order] = y_creation_data[value_order]
                    y_set_order += 1
                if y_set_order == buyer_number * seller_number:
                    y_data_set_order += 1
                    if y_data_set_order <= (data_set_creation_number * proportion):
                        train_y_set.append(y_set.reshape(buyer_number, seller_number))
                    else:
                        test_y_set.append(y_set.reshape(buyer_number, seller_number))
                    y_set_order = 0

    """
    plot图像化显示生成的数据，查看生成数据的性质
    """
    # for data_set_order_order in range(x_data_set_order):
    #     plt.figure(figsize=(8, 6), dpi=100)
    #     plt.hist(train_x_set[data_set_order_order].flatten(), bins=10, density=True)
    #     x_forplot = np.linspace(-5, 5, 1000)
    #     plt.plot(x_forplot, norm.pdf(x_forplot, loc=mu_init_x, scale=sigma_init_x), linewidth=3)
    #     plt.xlabel("the value of x_data", fontsize=10)
    #     plt.ylabel("x_data frequency", fontsize=10)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.legend(['theoretical pdf', 'Bos-Muller sample frequencies'], fontsize=10)
    #     plt.show()
    print("正态分布数据：数据已生成并完成采样")
    sio.savemat('./mat_data/10_10/initial_x.mat', {'x': test_x_set[index * BATCH_SIZE].data.numpy()})
    sio.savemat('./mat_data/10_10/initial_y.mat', {'y': test_y_set[index * BATCH_SIZE].data.numpy()})
    return train_x_set, train_y_set, test_x_set, test_y_set


def test_creation_data_x_y():
    train_x_set, train_y_set, test_x_set, test_y_set = creation_data_x_y()
    print("train_x_set的样本数：", len(train_x_set))
    print("train_y_set的样本数：", len(train_y_set))
    print("test_x_set的样本数：", len(test_x_set))
    print("test_y_set的样本数：", len(test_y_set))
    print("其中一个train_x_set样本的均值", train_x_set[150].mean())
    print("其中一个train_y_set样本的均值", train_y_set[150].mean())
    print("其中一个test_x_set样本的均值", test_x_set[150].mean())
    print("其中一个test_y_set样本的均值", test_y_set[150].mean())
    print("其中一个train_x_set样本的方差", train_x_set[150].std())
    print("其中一个train_y_set样本的方差", train_y_set[150].std())
    print("其中一个test_x_set样本的方差", test_x_set[150].std())
    print("其中一个test_y_set样本的方差", test_y_set[150].std())



