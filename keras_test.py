import numpy as np
import os
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.layers import Dropout,Embedding
from keras.layers import LSTM,Reshape
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from add_batch_size_test import preprocessing_data
from keras.utils import to_categorical
if __name__ == '__main__':
    # 定义超参
    learning_rate = 0.0001
    batch_size = 30
    seq_max_lenth = 300
    n_input = 3
    n_hidden_units_dense = 64
    n_hidden_units_lstm = 128
    n_classes = 2
    Epoch = 50
    model_dir = "model/lstm_model"
    model_prefix = "lstm"

    input_data, new_label, sequence_lenth = preprocessing_data()
    batch_X = np.array(input_data)
    batch_y = to_categorical(new_label, num_classes=n_classes)
    batch_seq = sequence_lenth

    """建立模型"""
    print("Building Model......")
    model = Sequential()
    # model.add(Reshape(( n_input,), input_shape=(seq_max_lenth,n_input)))
    model.add(Dense(n_hidden_units_dense,input_shape=(seq_max_lenth,n_input)))
    # model.add(Reshape((seq_max_lenth,n_hidden_units_dense), input_shape=(n_input,)))
    model.add(LSTM(n_hidden_units_lstm))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    # earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    tensorboard = TensorBoard(log_dir="model/log")
    checkpoint_path = "model/best_weights.hdf5"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, monitor="val_loss", verbose=1)
    model.fit(batch_X, batch_y, batch_size=batch_size, epochs=Epoch, callbacks=[cp_callback, tensorboard])
    print(model.summary())