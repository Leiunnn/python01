import numpy as np
import pandas as pd
import os


def read_data():
    # 读取数据
    data_output_imag = pd.read_csv(os.path.join('RxDistortData','5dB_sourcedata_imag.csv'), header=None)
    data_output_real = pd.read_csv(os.path.join('RxDistortData','5dB_sourcedata_real.csv'), header=None)
    data_input_imag = pd.read_csv(os.path.join('RxDistortData','5dB_inputdata_imag.csv'), header=None)
    data_input_real = pd.read_csv(os.path.join('RxDistortData','5dB_inputdata_real.csv'), header=None)


    # 分开输入输出,实部虚部放一起
    X = pd.concat([data_input_real, data_input_imag], axis=0)
    Y = pd.concat([data_output_real, data_output_imag], axis=0)

    return np.array(X), np.array(Y)


def data_choose(totallength_X, totallength_Y, length):

    rand_index_check = np.random.choice(len(totallength_X), size=length)
    Output_X = totallength_X[:, rand_index_check]
    Output_Y = totallength_Y[:, rand_index_check]

    return Output_X,Output_Y