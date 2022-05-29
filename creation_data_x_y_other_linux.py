#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
自定义生成买家卖家的报价数据，并且保存起来，生成的数据以7:3或者8:2的比例划分成train_set,test_set
data_set_order: 需要的数据集组数
buyer_number: 买家的人数
seller_number: 卖家的人数
data_set_order： 数据集的编号
x_set_name: 买家报价数据集的文件名
y_set_name: 卖家报价数据集的文件名
x_set: 随机生成的买家报价数据集
y_set: 随机生成的卖家报价数据集
x_set_data：x_set的tensor转成numpy
y_set_data: y_set的tensor转成numpy
2020/12/30
...
"""
from parameter_Iter_BAP_linux import *


"""
随机生成20000组数据，买家10000组报价数据x， 卖家10000组报价数据，同时要实现报价数据保存在对应的list中，并且为了查看和以后重复利用这些数据，
这些数据还要保存在txt文件中，且要命名规范，保证后续数据的提取。
"""


def creation_data_x_y_other():
    train_x_set = []
    train_y_set = []
    test_x_set = []
    test_y_set = []
    for data_set_order in range(data_set_creation_number):
        x_set_name = "x_set_%d" % data_set_order
        y_set_name = "y_set_%d" % data_set_order
        # x_set = torch.rand(buyer_number, seller_number)     # 第一种生成方式：生成随机数范围(0, 1)
        # y_set = torch.rand(seller_number, buyer_number)

        x_set = 7 * torch.rand(buyer_number, seller_number)     # 第二种生成方式：生成x随机数范围(0, 7), y随机数范围(0, 2)
        y_set = 2 * torch.rand(seller_number, buyer_number)
        # x_set = 7 * torch.rand(buyer_number, seller_number)     # 第三种生成方式：生成x随机数范围(0, 7), y随机数范围(0, 2)
        # y_set = 2 * torch.rand(buyer_number, seller_number)

        # x_set = random.uniform(0, 7, size=(buyer_number, seller_number))        # 第四种生成方式：生成x均匀随机数范围(0, 7), y均匀随机数范围(0, 2)
        # x_set = torch.from_numpy(x_set)
        # y_set = random.uniform(0, 2, size=(buyer_number, seller_number))
        # y_set = torch.from_numpy(y_set)

        # x_set = random.random(0, 7, size=(buyer_number, seller_number))        # 第四种生成方式：生成x随机数范围(0, 7), y随机数范围(0, 2)
        # x_set = torch.from_numpy(x_set)
        # y_set = random.random(0, 2, size=(buyer_number, seller_number))
        # y_set = torch.from_numpy(y_set)

        x_set_data = x_set.data.numpy()
        y_set_data = y_set.data.numpy()
        set_name_list = [x_set_name, y_set_name]
        set_list = [x_set_data, y_set_data]
        if (data_set_order + 1) <= (data_set_creation_number * proportion):
            train_x_set.append(x_set)
            train_y_set.append(y_set)
        else:
            test_x_set.append(x_set)
            test_y_set.append(y_set)
        """
        为什么买家数据写入和卖家数据写入要分开，是为了之后的实现买卖双方人数不对等的时候，程序同样可以使用，
        所以现在要把所有的buyer_number看做不等于seller_number
        """
        filename_1 = open("./creation_data/10_10/" + set_name_list[0] + ".txt", "w")
        for value in set_list[0]:
            for value_obtain_order in range(buyer_number):
                filename_1.write(str(value[value_obtain_order]))
                filename_1.write(" ")
                if value_obtain_order == buyer_number - 1:
                    filename_1.write("\n")
        filename_1.close()
        filename_2 = open("./creation_data/10_10/" + set_name_list[1] + ".txt", "w")
        for value in set_list[1]:
            for value_obtain_order in range(seller_number):
                filename_2.write(str(value[value_obtain_order]))
                filename_2.write(" ")
                if value_obtain_order == seller_number - 1:
                    filename_2.write("\n")
        filename_2.close()
    return train_x_set, train_y_set, test_x_set, test_y_set









