from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

class Data_process():
    def __init__(self,data_path,data_names,has_label=True):
        self.data_path = data_path
        self.data_names = data_names
        self.has_label = has_label
        self.original_data = pd.read_table(self.data_path, sep=" ", names=self.data_names)

        # 利用np.random.seed ==>设置种子数，np.random.shuffle ==>打乱数据
        if self.has_label:
            self.label = np.array(self.original_data["label"])
            self.move_data_all = np.array(self.original_data["move_data"])
            self.target_point = np.array(self.original_data["target"])
            np.random.seed(110)
            np.random.shuffle(self.label)
            np.random.seed(110)
            np.random.shuffle(self.move_data_all)
            np.random.seed(110)
            np.random.shuffle(self.target_point)
        else:
            self.move_data_all = np.array(self.original_data["move_data"])
            self.target_point = np.array(self.original_data["target"])
            np.random.seed(110)
            np.random.shuffle(self.move_data_all)
            np.random.seed(110)
            np.random.shuffle(self.target_point)


    def scale_exept_0(self,arr,seq_lenth):
        cols = arr.shape[-1]
        rows = arr.shape[0]
        new_arr = []
        for i in range(cols):
            # 每一列进行scale操作
            col_arr = arr[:, i][:seq_lenth]
            # 排序
            sort_col_arr = sorted(col_arr, reverse=True)
            new_col = []
            max_min = sort_col_arr[0] - sort_col_arr[-1]
            for ii in range(len(arr[:, 1])):
                if ii < seq_lenth:
                    if max_min > 0:
                        new_col.append((arr[:,i][ii] - sort_col_arr[-1]) / max_min)
                    else:
                        new_col.append(1)
                else:
                    new_col.append(0)
            new_arr.append(new_col)
        new_arr = np.array(new_arr).transpose()
        return new_arr

    def preprocessing_data(self):
        # 用字典的形式来存储move_data每一个用户的n个点
        move_data_dict = {}
        for i in range(len(self.move_data_all)):
            move_data_dict.setdefault(i, {})
            # 对于每一行来说，都是str，用split进行分割
            move_data_single = self.move_data_all[i].split(";")
            for ii in range(len(move_data_single)):
                if move_data_single[ii] != '':
                    move_data_dict[i].setdefault(ii, {})
                    move_data_dict[i][ii] = move_data_single[ii]
        # print(move_data_dict)
        # 查看下move_data中最长的move轨迹有多少个点
        single_user_len = []
        for key, value in move_data_dict.items():
            single_len = len(value)
            single_user_len.append(single_len)
        # max_single_user_len = sorted(single_user_len, reverse=True)[0]
        max_single_user_len = 300
        print("The laggest move data is {}".format(max_single_user_len))  # 最长300
        # 对于后面放进lstm进行学习的时候，必须要将每一个用户移动轨迹的点的长度固定（这里设置0，0，0 来对movedata少的用户轨迹进行填充）
        new_move_dict = {}
        self.sequence_lenth = []
        for user, move_data in move_data_dict.items():
            self.sequence_lenth.append(len(move_data))
            new_move_dict.setdefault(user, {})
            if len(move_data) < max_single_user_len:
                for i in range(max_single_user_len):
                    if i < len(move_data):
                        new_move_dict[user][i] = move_data[i]
                    else:
                        new_move_dict[user][i] = '0,0,0'
            else:
                new_move_dict[user] = move_data
        # print(new_move_dict)
        # 将move_data 转成(3000,300,3)的数组
        self.input_data = []
        for user, user_value in new_move_dict.items():
            user_list = []
            for move_index, move_value in new_move_dict[user].items():
                every_move = move_value.split(',')
                move_x = []
                for every_move_s in every_move:
                    move_x.append(int(every_move_s))
                user_list.append(move_x)
            np_user_list = np.array(user_list).astype(np.float32)
            self.input_data.append(np_user_list)
        # 看下正负样本比例(正样本应该是异常点：少量，负样本应该是正常点的：多量)
        if self.has_label:
            self.new_label = []
            for i in self.label:
                if i == 1:
                    self.new_label.append(0)
                else:
                    self.new_label.append(1)
            return self.input_data, self.new_label, self.sequence_lenth
        else:
            return self.input_data,self.sequence_lenth


