import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from datetime import datetime
import os
import sklearn as sk
from sklearn.metrics import confusion_matrix,recall_score,precision_score,f1_score,roc_curve

class LSTM_model():
    def __init__(self,input_data=None,label=None,sequence_lenth=None,val_X = None,val_y = None,val_seq = None):
        # 定义超参
        self.learning_rate = 0.005
        self.batch_size = 100
        self.seq_max_lenth = 300
        self.n_input = 7
        self.n_hidden_units_dense = 64
        self.n_hidden_units_lstm = 128
        self.n_classes = 2
        self.Epoch = 1000
        self.attention_size = 256
        self.model_dir = "model/lstm_model"
        self.model_prefix = "lstm"

        # 训练集
        self.input_data = input_data
        self.label = label
        self.sequence_lenth = sequence_lenth
        self.batch_X = np.array(self.input_data)
        if self.label:
            self.batch_y = to_categorical(self.label, num_classes=self.n_classes)
        self.batch_seq = self.sequence_lenth
        if self.input_data:
            self.all_sample_num = len(self.input_data)

        # 测试集
        self.val_X = val_X
        if val_y:
            self.val_y = to_categorical(val_y, num_classes=self.n_classes)
            self.val_y_orig = val_y
        self.val_seq = val_seq
        if self.val_X:
            self.all_val_num = len(self.val_X)

        # 定义placeholder
        self.Xs = tf.placeholder(tf.float32, shape=(None, self.seq_max_lenth, self.n_input))
        self.ys = tf.placeholder(tf.float32, shape=(None, self.n_classes))
        self.seq_len = tf.placeholder(tf.int32, shape=(None))

    def focal_loss(self,labels, logits, gamma=2):
        """
        Computer focal loss for multi classification
        Args:
          labels: A int32 tensor of shape [batch_size,num_classes].
          logits: A float32 tensor of shape [batch_size,num_classes].
          gamma: A scalar for focal loss gamma hyper-parameter.
        Returns:
          A tensor of the same shape as `lables`
        """
        y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
        # labels = tf.one_hot(labels, depth=y_pred.shape[1])
        L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
        L = tf.reduce_sum(L, axis=1)
        return L

    def focal_loss_2(self, onehot_labels, cls_preds, alpha=0.25, gamma=2.0, name=None, scope=None):
        with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
            logits = tf.convert_to_tensor(cls_preds)
            onehot_labels = tf.convert_to_tensor(onehot_labels)

            precise_logits = tf.cast(logits, tf.float32) if (
                    logits.dtype == tf.float16) else logits
            onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
            predictions = tf.nn.sigmoid(logits)
            predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1. - predictions)
            # add small value to avoid 0
            epsilon = 1e-8
            alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
            alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1 - alpha_t)
            losses = tf.reduce_sum(
                -alpha_t * tf.pow(1. - predictions_pt, gamma) * onehot_labels * tf.log(predictions_pt + epsilon),
                name=name, axis=1)
            return losses

    def add_layer(self,inputs,in_size,out_size,activation_function = None):
        weights = tf.Variable(tf.truncated_normal([in_size, out_size], mean=0.0, stddev=1.0, dtype=tf.float32))
        # weights = tf.get_variable("w11111",shape=[in_size,out_size], initializer=tf.contrib.layers.xavier_initializer())
        baise = tf.Variable(tf.constant(0.1, shape=[out_size, ]))
        result = tf.add(tf.matmul(inputs, weights), baise)
        if activation_function is None:
            outputs = result
        else:
            outputs = activation_function(result)
        return outputs

    def add_bidirectional_LSTM_layer(self,X,n_hidden_units,batch_size,sequence_length):
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

    def add_LSTM_layer(self,X,n_hidden_units,batch_size,sequence_length):
        # 创建一个lstm细胞核
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
        # 定义一个初始状态
        init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X, initial_state=init_state,
                                                 sequence_length=sequence_length)
        return final_state[1]

    def attention(self,inputs, attention_size, time_major=False, return_alphas=False):

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        w2_omega = tf.Variable(tf.random_normal([attention_size,attention_size],stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)
            v2 = tf.tanh(tf.tensordot(v,w2_omega,axes=1)+b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v2, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def model_building(self,Xs,seq_len,bidirection_lstm_layer=False,attention_layer=False):
        """用lstm建模（层数：input(30,300,7) ==> ful_collected_layer(30,300,64) ==> lstm ==> ful_collected_layer ==> output）"""
        # reshape X(batch_size,seq_max_lenth,n_input) ==>(bath_size*seq_max_lenth,n_input)
        Xs_reshaped = tf.reshape(Xs, [-1, self.n_input])
        # the First Dense layer
        self.dense_layer1 = self.add_layer(Xs_reshaped,self.n_input,self.n_hidden_units_dense,tf.nn.leaky_relu)
        # reshape to 3D(bath_size,seq_max_lenth,n_hidden_units)
        self.dense1_result_reshaped = tf.reshape(self.dense_layer1, [-1, self.seq_max_lenth, self.n_hidden_units_dense])
        # building the LSTM layer,output ==> (batch_size,n_hidden_)(30,128)

        """这里如果是加入双向lstm"""
        if bidirection_lstm_layer:

            self.lstm_output, self.final_state_h = self.add_bidirectional_LSTM_layer(self.dense1_result_reshaped,
                                                                                     self.n_hidden_units_lstm,
                                                                                     self.batch_size, seq_len)

            if attention_layer:
                """对lstm_output加入attention机制"""
                self.attention_lstm_output = self.attention(self.lstm_output, self.attention_size)
                self.dense_layer2 = self.add_layer(self.attention_lstm_output, self.attention_size,
                                                   self.n_hidden_units_dense,
                                                   tf.nn.tanh)
                self.dense_layer3 = self.add_layer(self.dense_layer2, self.n_hidden_units_dense, self.n_classes,
                                                   tf.nn.softmax)
                return self.dense_layer3
            else:
                # Second Dense layer(这里不要加上softmax，因为在后面定义softmax_cross_entropy_with_logits里面tf封装好了)
                # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys,logits=prediction))
                self.dense_layer2 = self.add_layer(self.final_state_h, self.n_hidden_units_lstm * 2,
                                                   self.n_hidden_units_lstm, tf.nn.tanh)
                self.dense_layer3 = self.add_layer(self.dense_layer2, self.n_hidden_units_lstm, self.n_classes,
                                                   tf.nn.softmax)
                return self.dense_layer3
        else:
            self.lstm_output = self.add_LSTM_layer(self.dense1_result_reshaped, self.n_hidden_units_lstm, self.batch_size, seq_len)
            # Second Dense layer(这里不要加上softmax，因为在后面定义softmax_cross_entropy_with_logits里面tf封装好了)
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ys,logits=prediction))
            self.dense_layer2 = self.add_layer(self.lstm_output,self.n_hidden_units_lstm,self.n_hidden_units_lstm,tf.nn.tanh)
            self.dense_layer3 = self.add_layer(self.dense_layer2, self.n_hidden_units_lstm, self.n_classes, tf.nn.softmax)
            return self.dense_layer3

    def model_eval(self,ys, focal_loss=False):
        with tf.name_scope("Metrics"):
            """定义loss"""
            if focal_loss:
                self.cross_entropy = tf.reduce_mean(self.focal_loss(labels= ys,logits=self.dense_layer3))
                tf.summary.scalar("loss",self.cross_entropy)
            else:
                # 这里用自己写的交叉熵，这里用到了数值的裁剪（放进log中的值必须大于0）
                self.cross_entropy = tf.reduce_mean(
                    -tf.reduce_sum(ys * tf.log(tf.clip_by_value(self.dense_layer3, 1e-10, 1.0))))
                tf.summary.scalar("loss", self.cross_entropy)
                # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)

            """梯度剪裁"""
            opt = tf.train.AdadeltaOptimizer(self.learning_rate)
            # Compute the gradients for a list of variables.
            grads_and_vars = opt.compute_gradients(self.cross_entropy, tf.trainable_variables())
            # grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
            # need to the 'gradient' part, for example cap them, etc.
            capped_grads_and_vars = [(tf.clip_by_value(gv[0], 0.1, 5.), gv[1]) for gv in grads_and_vars]
            # Ask the optimizer to apply the capped gradients.
            self.optimizer_train = opt.apply_gradients(capped_grads_and_vars)

            """定义准确性"""

            correct_pred = tf.equal(tf.arg_max(self.dense_layer3, 1), tf.arg_max(ys, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

    def train_val(self):
        '''
        run model函数主要是控制整个训练的流程，需要传入session，调用session.run(variables)会得到variables里面各个变量的值。
        这里当训练模式的时候，也就是training!=None，我们传入的training是之前定义的train_op，调用session.run(train_op)会自动完成反向求导，
        整个模型的参数会发生更新。当training==None时,是我们需要对验证集合做一次预测的时候(或者是测试阶段)，
        这时我们不需要反向求导，所以variables里面并没有加入train_op
        '''
        # define self.saver
        self.saver = tf.train.Saver(tf.global_variables())
        self.init_op = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter("model/log/train",self.accuracy.graph)
        test_writer = tf.summary.FileWriter("model/log/test", self.accuracy.graph)
        """构建会话"""
        with tf.Session() as sess:
            sess.run(self.init_op)
            # restore the last self.checkpoint
            start_epoch = 0
            self.checkpoint = tf.train.latest_checkpoint(self.model_dir)
            if self.checkpoint:
                self.saver.restore(sess, self.checkpoint)
                print("Restoring from the self.checkpoint {}".format(self.checkpoint))
                start_epoch += int(self.checkpoint.split('-')[1])
            print("Start trianing...")
            try:
                for epoch in range(start_epoch, self.Epoch):
                    start_time = datetime.now()
                    n = 0
                    acc_all = []
                    loss_all = []
                    n_chunk = self.all_sample_num // self.batch_size
                    for batch in range(n_chunk):
                        if self.batch_size * (n + 1) <= self.all_sample_num:
                            now_time_start = datetime.now()
                            batch_X_single = self.batch_X[self.batch_size * n:self.batch_size * (n + 1)]
                            batch_y_single = self.batch_y[self.batch_size * n:self.batch_size * (n + 1)]
                            batch_seq_single = self.batch_seq[self.batch_size * n:self.batch_size * (n + 1)]
                            summary_bs, optimizer_bs = sess.run([merged, self.optimizer_train],
                                                                feed_dict={self.Xs: batch_X_single,
                                                                           self.ys: batch_y_single,
                                                                           self.seq_len: batch_seq_single})
                            loss_bs, accuracy_bs = sess.run([self.cross_entropy, self.accuracy],
                                                            feed_dict={self.Xs: batch_X_single,
                                                                       self.ys: batch_y_single,
                                                                       self.seq_len: batch_seq_single})
                            train_writer.add_summary(summary_bs,epoch*batch)
                            n += 1
                            now_time = datetime.now()
                            print('Epoch: {}, batch: {},acc: {:.2%},loss: {:.4f} using time : {}'.format(
                                epoch + 1, batch + 1, accuracy_bs, loss_bs, now_time - now_time_start))
                            acc_all.append(accuracy_bs)
                            loss_all.append(loss_bs)
                    # auto save self.checkpoint
                    if epoch % 10 == 0:
                        self.saver.save(sess, os.path.join(self.model_dir, self.model_prefix), global_step=epoch)
                    end_time = datetime.now()
                    print("{} Epoch has time {},mean of acc is {:.2%},and mean of loss is {:.4f}".format(epoch,
                                                                                                         end_time - start_time,
                                                                                                         np.array(
                                                                                                             acc_all).mean(),
                                                                                                         np.array(
                                                                                                             loss_all).mean()))
                    nn = 0
                    acc_val = []
                    loss_val = []
                    pred_val_all = []
                    n_chunk_val =self.all_val_num // self.batch_size
                    for batch_val in range(n_chunk_val):
                        if self.batch_size * (nn + 1) <= self.all_val_num:
                            val_X_single = self.val_X[self.batch_size * nn:self.batch_size * (nn + 1)]
                            val_y_single = self.val_y[self.batch_size * nn:self.batch_size * (nn + 1)]
                            val_seq_single = self.val_seq[self.batch_size * nn:self.batch_size * (nn + 1)]
                            # tf.argmax(dimension = 1)是按照行来找
                            loss_v, accuracy_v,summary_v,pred_v = sess.run([self.cross_entropy, self.accuracy,merged,tf.arg_max(self.dense_layer3,1)],
                                                            feed_dict={self.Xs: val_X_single,
                                                                       self.ys: val_y_single,
                                                                       self.seq_len: val_seq_single})
                            test_writer.add_summary(summary_v,epoch*batch)
                            nn += 1
                            acc_val.append(accuracy_v)
                            loss_val.append(loss_v)
                            pred_val_all.append(pred_v)
                    print("Val input result : mean of val_acc is {:.2%},"
                          "and mean of val_loss is {:.4f} \n".format(np.array(acc_val).mean(),np.array(loss_val).mean()))
                    print("Confusion Matrix:")
                    y_test_pred_all = pred_val_all
                    y_test_pred = []
                    for i in range(len(y_test_pred_all)):
                        for ii in range(len(y_test_pred_all[i])):
                            y_test_pred.append(y_test_pred_all[i][ii])
                    y_test_true = self.val_y_orig
                    print(confusion_matrix(y_test_true, y_test_pred))
                    print("Precision", precision_score(y_test_true, y_test_pred))
                    print("Recall", recall_score(y_test_true, y_test_pred))
                    print("f1_score", f1_score(y_test_true, y_test_pred))
                    fpr, tpr, tresholds = roc_curve(y_test_true, y_test_pred)


            except KeyboardInterrupt:
                print("Interrupt manually,try saving self.checkpoint from now...")
                self.saver.save(sess, os.path.join(self.model_dir, self.model_prefix), global_step=epoch)

    def prediction(self,input_feature,seq_len):
        tf.reset_default_graph()
        input_feature = np.array(input_feature).reshape([1, self.seq_max_lenth, self.n_input])

        """开始加载模型管道"""
        # 这里在bidirectionRNN中sequence_lenth参数是一个list，而如果seq_len输出是一个单纯的int，会报错
        seq_len_list = []
        seq_len_list.append(seq_len)
        prediction = self.model_building(self.Xs,self.seq_len,bidirection_lstm_layer=True)
        """加载训练好的模型参数"""
        saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            checkpoint = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, checkpoint)
            result = sess.run(prediction, feed_dict={self.Xs: input_feature, self.seq_len: seq_len_list})
            print(result)
            return result



