import numpy as np
import pandas as pd
import tensorflow as tf

"""测试dynamic rnn的返回值到底是什么"""
def dynamic_rnn(rnn_type='lstm'):
    # 创建输入数据,3代表batch size,6代表输入序列的最大步长(max time),8代表每个序列的维度
    X = np.random.randn(3, 6, 4)

    # 第二个输入的实际长度为4
    X[1, 4:] = 0

    # 记录三个输入的实际步长
    X_lengths = [6, 4, 6]

    rnn_hidden_size = 5
    if rnn_type == 'lstm':
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.GRUCell(num_units=rnn_hidden_size)

    outputs, last_states = tf.nn.dynamic_rnn(
        cell=cell,
        dtype=tf.float64,
        sequence_length=X_lengths,
        inputs=X)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        o1, s1 = session.run([outputs, last_states])
        print(np.shape(o1))
        print(o1)
        print(np.shape(s1))
        print(s1)

"""Tensorflow 中矩阵(shape = [28,128])和向量(shape = [128,])相加"""
def test1():
    a = tf.Variable(tf.constant([1, 2, 3, 4, 5], shape=[1, 5]))
    b = tf.Variable(tf.constant(2, shape=[3, 5]))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print(sess.run(a))
        print(sess.run(b))
        print(sess.run(tf.add(b, a)))


def scale_exept_0(arr,seq_lenth):
    cols = arr.shape[-1]
    rows = arr.shape[0]
    new_arr = []
    for i in range(cols):
        # 每一列进行scale操作
        col_arr = arr[:,i]
        # 排序
        sort_col_arr = sorted(col_arr,reverse=True)
        new_col = []
        max = sort_col_arr[0]
        if sort_col_arr[-1] == 0:
            max_min = max - sort_col_arr[seq_lenth-1]
            for ii in range(len(col_arr)):
                if ii < seq_lenth:
                    new_col.append(col_arr[ii]/max_min)
                else:
                    new_col.append(0)

        else:
            max_min = max - sort_col_arr[-1]
            for ii in range(len(col_arr)):
                if ii < seq_lenth:
                    new_col.append(col_arr[ii]/max_min)
                else:
                    new_col.append(0)
        new_arr.append(new_col)
    new_arr = np.array(new_arr).transpose()
    return new_arr

def attention(inputs, attention_size, time_major=False, return_alphas=False):

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

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):

    '''
    run model函数主要是控制整个训练的流程，需要传入session，调用session.run(variables)会得到variables里面各个变量的值。
    这里当训练模式的时候，也就是training!=None，我们传入的training是之前定义的train_op，调用session.run(train_op)会自动完成反向求导，
    整个模型的参数会发生更新。
    当training==None时,是我们需要对验证集合做一次预测的时候(或者是测试阶段)，这时我们不需要反向求导，所以variables里面并没有加入train_op
    '''
    # have tensorflow compute accuracy
    # 计算准确度（ACC值）
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    # 对训练样本进行混洗
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    # 设置需要计算的变量
    # 如果需要进行训练，将训练过程(training)也加进来
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training

    # counter
    # 进行迭代
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        # 记录损失函数和准确度的变化
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        # 确保每个训练样本都被遍历
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            # 产生一个minibatch的样本
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            # 生成一个输入字典(feed dictionary)
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            # 获取minibatch的大小
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            # 计算损失函数和准确率
            # 如果是训练模式的话，执行训练过程
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            # aggregate performance stats
            # 记录本轮的训练表现
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            # 定期输出模型表现
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct


if __name__ == '__main__':
    a = tf.constant([[[1,2,3,4],
                      [2,3,4,5],
                      [3,4,5,6]],
                     [[1, 2, 3, 4],
                      [2, 3, 4, 5],
                      [3, 4, 5, 6]]
                     ])
    b = tf.constant([[1,2,3],
                      [2,3,4]])
    c = tf.expand_dims(b,-1)
    d = a*c
    e = tf.reduce_sum(d,1)
    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
            sess.run(tf.global_variables_initializer())
            print(sess.run(c))
            print(sess.run(d))
            print(sess.run(e))
