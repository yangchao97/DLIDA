#!/home/zhengyangchao/anaconda3/envs/pytorch_env/bin/python
"""
本code只是用来画X数据的散点图
2022/04/13 by yangchao97
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn

with open('./objs_data/10_10/objs.pkl', 'rb') as objs_file:
    init_test_x_and_y, train_x_set, train_y_set, model_save_path = pickle.load(objs_file)
device = torch.device("c"
                      "uda:1" if torch.cuda.is_available() else "cpu")
x_list = init_test_x_and_y[0][0].cpu()


def x_data_scatter():
    for i in range(9):
        x_value = x_list[i-1]
        y_value = [i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1, i+1]
        plt.scatter(y_value, x_value, s=20, c="#ff1212", marker='o')
        plt.xlabel('row')
        plt.ylabel('value')

    x_value = x_list[9]
    y_value = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    my_x_ticks_10 = np.arange(1, 11, 1)
    plt.xticks(my_x_ticks_10)
    plt.scatter(y_value, x_value, s=20, c="#ff1212", marker='o', label="x_data")
    plt.xlabel('row')
    plt.ylabel('value')
    plt.legend(loc="best")
    plt.show()


x_data_scatter()
