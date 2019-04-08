import numpy as np
import pandas as pd
import math
from data_process import Data_process
from lstm_model import LSTM_model

def feature_engineering(original_data_process,input_data,sequence_lenth):
    target_point = original_data_process.target_point
    target_x_all = []
    target_y_all = []
    for i in range(len(target_point)):
        target_x = target_point[i].split(",")[0]
        target_y = target_point[i].split(",")[1]
        target_x_all.append(float(target_x))
        target_y_all.append(float(target_y))
    target_point_new = np.array(pd.DataFrame({"x": target_x_all, "y": target_y_all}))
    print(target_point_new)

    """1.每个点的速度"""
    new_input_data = []
    for i in range(len(input_data)):
        trace = input_data[i]
        target_single = target_point_new[i]
        seq_single = sequence_lenth[i]
        target_single_x = target_single[0]
        target_single_y = target_single[1]
        # 1.速度特征
        velocity = []
        # 2.与前一个点的距离
        distance_diff_expoint = []
        # 3.与前一个点的时间
        time_diff_expoint = []
        # 4.与前一个点的加速度
        acceleration = []
        # 5.与前一个点的角度差
        theta_diff_expoint = []
        # 6.与目标点的距离
        distance_diff_target = []
        # 7.与目标点的角度
        theta_diff_target = []

        for ii in range(len(trace)):
            trace_x = trace[ii, 0]
            trace_y = trace[ii, 1]
            trace_t = trace[ii, 2]
            if ii == 0:
                velocity.append(0)
                distance_diff_expoint.append(0)
                time_diff_expoint.append(0)
                acceleration.append(0)
                theta_diff_expoint.append(0)
                distance_diff_target.append(
                    math.sqrt((trace_x - target_single_x) ** 2 + (trace_y - target_single_y) ** 2))
                theta_diff_target.append(math.atan((trace_y - target_single_y) / (trace_x - target_single_x)))
            elif ii > 0 and ii < seq_single:
                distance_1_2 = math.sqrt((trace_x - trace[ii - 1, 0]) ** 2 + (trace_y - trace[ii - 1, 1]) ** 2)
                time_1_2 = trace_t - trace[ii - 1, 2]
                # 存在两个点之间的时间差为0的情况，就用99来填充
                if time_1_2 == 0:
                    velocity.append(99)
                    acceleration.append(99)
                else:
                    velocity.append(distance_1_2 / time_1_2)
                    acceleration.append(distance_1_2 / (time_1_2 ** 2))
                distance_diff_expoint.append(distance_1_2)
                time_diff_expoint.append(time_1_2)
                if trace_x - trace[ii - 1, 0] == 0:
                    theta_diff_expoint.append(math.atan(99))
                else:
                    theta_diff_expoint.append(math.atan((trace_y - trace[ii - 1, 1]) / (trace_x - trace[ii - 1, 0])))
                distance_diff_target.append(
                    math.sqrt((trace_x - target_single_x) ** 2 + (trace_y - target_single_y) ** 2))
                theta_diff_target.append(math.atan((trace_y - target_single_y) / (trace_x - target_single_x)))
            else:
                velocity.append(0)
                distance_diff_expoint.append(0)
                time_diff_expoint.append(0)
                acceleration.append(0)
                theta_diff_expoint.append(0)
                distance_diff_target.append(0)
                theta_diff_target.append(0)
        feature = np.array(
            pd.DataFrame({"1": velocity, "2": distance_diff_expoint, "3": time_diff_expoint, "4": acceleration,
                          "5": theta_diff_expoint, "6": distance_diff_target, "7": theta_diff_target}))
        new_input_data.append(feature)
    # print(new_input_data)

    # 对new_input_data 进行标准化
    scale_input_data = []
    for i in range(len(new_input_data)):
        seq_lenth_single = sequence_lenth[i]
        array_single = new_input_data[i]
        new_arr = original_data_process.scale_exept_0(array_single, seq_lenth_single)
        scale_input_data.append(new_arr)
    return scale_input_data


if __name__ == '__main__':
    input_path = "data/dsjtzs_txfz_training.txt"
    input_names = ["index", "move_data", "target", "label"]
    original_data_process = Data_process(input_path,input_names)
    input_data, label, sequence_lenth = original_data_process.preprocessing_data()
    """划分训练集和测试集，由于之前已经shuffle过了，直接取前2500条数据作为训练集，500条作为测试集"""
    # 做完特征后的数据
    input_feature = feature_engineering(original_data_process,input_data,sequence_lenth)
    trainX = input_feature[:2500]
    testX = input_feature[2500:]
    trainY = label[:2500]
    testY = label[2500:]
    sequence_lenth_train = sequence_lenth[:2500]
    sequence_lenth_test = sequence_lenth[2500:]

    # 检查下scale_input_data 有没有nan值
    list_nan = list(map(tuple, np.argwhere(np.isnan(np.array(input_feature)))))
    print(list_nan)

    lstm_model = LSTM_model(input_data=trainX,sequence_lenth = sequence_lenth_train,label = trainY,val_X=testX,val_y=testY,val_seq=sequence_lenth_test)
    # val_onehot_y = lstm_model.val_y
    # original_y = testY
    # 定义placeholder
    Xs = lstm_model.Xs
    ys = lstm_model.ys
    seq_len = lstm_model.seq_len
    # 构建模型
    result = lstm_model.model_building(Xs=Xs,seq_len=seq_len,bidirection_lstm_layer=True,attention_layer=False)
    lstm_model.model_eval(ys=ys,focal_loss=False)
    lstm_model.train_val()
