""""用训练好的模型进行对测试集进行输出结果"""
import numpy as np
import tensorflow as tf
from data_process import Data_process
from lstm_feature_engineer import feature_engineering



def add_layer(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean=0.0, stddev=1.0, dtype=tf.float32))
    # weights = tf.get_variable("w11111",shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer())
    baise = tf.Variable(tf.constant(0.1, shape=[out_size, ]))
    result = tf.add(tf.matmul(inputs, weights), baise)
    if activation_function is None:
        outputs = result
    else:
        outputs = activation_function(result)
    return outputs

def model_building(Xs,seq_len,seq_max_lenth,n_input,n_classes,n_hidden_units_dense,n_hidden_units_lstm,batch_size,bidirection_lstm_layer=False,attention_layer=False):
    """用lstm建模（层数：input(30,300,7) ==> ful_collected_layer(30,300,64) ==> lstm ==> ful_collected_layer ==> output）"""
    # reshape X(batch_size,seq_max_lenth,n_input) ==>(bath_size*seq_max_lenth,n_input)
    Xs_reshaped = tf.reshape(Xs, [-1, n_input])
    # the First Dense layer
    dense_layer1 = add_layer(Xs_reshaped,n_input,n_hidden_units_dense,tf.nn.leaky_relu)
    # reshape to 3D(bath_size,seq_max_lenth,n_hidden_units)
    dense1_result_reshaped = tf.reshape(dense_layer1, [-1, seq_max_lenth, n_hidden_units_dense])
    # building the LSTM layer,output ==> (batch_size,n_hidden_)(30,128)

    """这里如果是加入双向lstm"""
    if bidirection_lstm_layer:

        lstm_output, final_state_h = add_bidirectional_LSTM_layer(dense1_result_reshaped,
                                                                                 n_hidden_units_lstm,
                                                                                 batch_size, seq_len)


        # Second Dense layer(这里不要加上softmax，因为在后面定义softmax_cross_entropy_with_logits里面tf封装好了)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys,logits=prediction))
        dense_layer2 = add_layer(final_state_h, n_hidden_units_lstm * 2,
                                           n_hidden_units_lstm, tf.nn.tanh)
        dense_layer3 = add_layer(dense_layer2, n_hidden_units_lstm, n_classes,
                                           tf.nn.softmax)
        return dense_layer3
    else:
        lstm_output = add_LSTM_layer(dense1_result_reshaped, n_hidden_units_lstm, batch_size, seq_len)
        # Second Dense layer(这里不要加上softmax，因为在后面定义softmax_cross_entropy_with_logits里面tf封装好了)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys,logits=prediction))
        dense_layer2 = add_layer(lstm_output,n_hidden_units_lstm,n_hidden_units_lstm,tf.nn.tanh)
        dense_layer3 = add_layer(dense_layer2, n_hidden_units_lstm, n_classes, tf.nn.softmax)
        return dense_layer3

def add_bidirectional_LSTM_layer(X,n_hidden_units,batch_size,sequence_length):
    # 构建两个细胞核和两个初始状态量
    lstm_1_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    lstm_2_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    init_state_1 = lstm_1_cell.zero_state(batch_size, dtype=tf.float32)
    init_state_2 = lstm_2_cell.zero_state(batch_size, dtype=tf.float32)

    ((outputs_fw, outputs_bw), (outputs_state_fw, outputs_state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_1_cell,
                                                             cell_bw=lstm_2_cell,
                                                             inputs=X,
                                                             initial_state_fw=init_state_1,
                                                             initial_state_bw=init_state_2,
                                                             sequence_length=sequence_length)
    output_concat = tf.concat((outputs_fw,outputs_bw),2)
    final_state_c = tf.concat((outputs_state_fw.c,outputs_state_bw.c),1)
    final_state_h = tf.concat((outputs_state_fw.h, outputs_state_bw.h), 1)
    return output_concat,final_state_h


def add_LSTM_layer(X,n_hidden_units,batch_size,sequence_length):
    # 创建一个lstm细胞核
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # 定义一个初始状态
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X, initial_state=init_state,
                                             sequence_length=sequence_length)
    return final_state[1]



def model_predict(input_feature,seq_len):
    tf.reset_default_graph()
    batch_size = 1
    seq_max_lenth = 300,
    n_input = 7,
    n_classes = 2,
    n_hidden_units_dense = 64,
    n_hidden_units_lstm = 128
    input_feature = np.array(input_feature).reshape(1,300,7)

    """开始加载模型管道"""
    Xs = tf.placeholder(tf.float32, shape=(None, 300, 7))
    seq_tf_len = tf.placeholder(tf.int32,shape=(None))
    # 这里在bidirection
    seq_len_list = []
    seq_len_list.append(seq_len)
    prediction = model_building(Xs = Xs,seq_len=seq_tf_len,seq_max_lenth = 300,n_input=7,n_classes = 2,n_hidden_units_dense=64,
                                n_hidden_units_lstm=128,batch_size=1,bidirection_lstm_layer=True)

    """加载训练好的模型参数"""
    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        checkpoint = tf.train.latest_checkpoint("model/lstm_model")
        saver.restore(sess,checkpoint)
        result = sess.run(prediction,feed_dict={Xs:input_feature,seq_tf_len:seq_len_list})
        print(result)
        return result





if __name__ == '__main__':
    input_names = ["index", "move_data", "target"]
    input_path = "data/dsjtzs_txfz_test_sample.txt"
    original_no_label = Data_process(input_path,input_names,has_label=False)
    input_data,input_seq_lenth = original_no_label.preprocessing_data()
    input_feature = feature_engineering(original_no_label,input_data,input_seq_lenth)
    for i in range(len(input_feature)):
        result = model_predict(input_feature[i],input_seq_lenth[i])
        print("第{}条记录有{:.2%}概率是异常记录！".format(i + 1, result[0][1]))

    # tf.reset_default_graph()
    # lstm_model = LSTM_model()
    # # 定义placeholder
    # Xs = lstm_model.Xs
    # ys = lstm_model.ys
    # seq_len = lstm_model.seq_len
    # prediction = lstm_model.model_building(Xs=Xs,seq_len=seq_len,bidirection_lstm_layer=True)
    # # 做完特征后的数据
    # input_feature = feature_engineering(original_no_label,input_data, input_seq_lenth)
    # for i in range(len(input_feature)):
    #     result = lstm_model.prediction(input_feature[i],input_seq_lenth[i])
    #     print("第{}条记录有{:.2%}概率是异常记录！".format(i+1, result[0][0]))

