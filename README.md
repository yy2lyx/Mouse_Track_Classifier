# 鼠标移动轨迹模型
>数据集：2017年中国高校计算机大赛的初赛数据集（鼠标移动轨迹数据3000条样本） 作为我们实验的数据。

## 1.整体结构
* 用keras来测试的（随便写的）==> keras_model.py
* 看下数据轨迹的样子 ==> show_trace.py
* 用xgboost对已经做好特征的数据进行实验 ==> xgboost_test.py
* 用双向lstm模型建模
	* (1) 数据预处理 ==> data_process.py
	* (2) 建立特征 ==> lstm_feature_engineer.py
	* (3) 构建lstm模型（包含attention，双向，focal loss）==> lstm_model.py
	* (4) 让模型预测 ==> lstm_model_prediction.py
* 文件夹data 
	* (1) 用xgboost训练已经做好特征工程的数据 ==> alltrain.csv 
	* (2) 用lstm模型使用的训练数据（3000条）==> dsjtzs_txfz_training.txt
* 文件夹model
	* (1) 用tensorboard 打开log（内涵有train和test）==> tensorboard --logdir=log
	* (2) lstm模型保存的checkpoint都在 ==> lstm_model
	* (3) 用xgboost训练的结果 ==> xg_model.model
## 2. 训练结果评分
* xgboost ==> 训练时间很快，acc = 0.99，recall = 1，f1_score = 0.99（当构建的特征足够完善的时候，用xgboost很高效）
* lstm_model 
	* (1) 经历过模型震荡，loss忽上忽下（幅度过大）==> 这里是模型在只经过一层神经元(dense)的时候无法解决异或问题和非线性问题 ，因此在分类层构建了2层隐藏层。
	* (2) 搭建了2层隐藏层之后，发现模型稳定了，但是loss先逐渐降低，然后降到最低点又缓慢升高，这里是用于学习率可能设置大了（lr = 0.01）,因此将lr = 0.002之后，模型逐渐稳定收敛，达到最低点也不会升高。
	* (3) 模型中出现test集的准确率 > train集的准曲率（test的loss < train的loss）原因：
		+ 第一个是由于模型的拟合能力太弱，数据量太少（train2500，test500）；
		+ 第二个原因可能是在test可以视为正式考试，而这张试卷正好是学生擅长的题目（train中包含了大量test）
	* (4) 模型几种情况的评分（LR= 0.005，模型都是收敛的）：
		+ 第一种：双向lstm + Focal loss ：acc = 0.755，loss = 0.18，P = 0.86，R = 0.86
		+ 第二种：双向lstm + Attention ：acc = 0.86，loss = 29，P = 0.90，R = 0.95
		+ 第三种：双向lstm  ：acc = 0.95，loss = 18，P = 0.96，R = 0.97
		+ 第四种：单层lstm ： acc = 0.937，loss = 20，P = 0.95，R = 0.977
## 3.模型和预处理需要注意的地方
* 预处理
	* （1）任何数据（包括时序）都需要做特征工程的，哪怕只给(x,y,t)也是需要做每一个点的状态特征的。
	* （2）数据需要shuffle，而且一定是给的数据集来进行划分train和test（同分布）
	* （3）做完特征之后，需要scale
	* （4）找时序数据的时候，注意下max_seq_len的最大值必须确定（这里是300）
* 模型预测
	* （1）先加载tensorflow的管道(model的结构)
	* （2）将数据处理的时候，注意下他们的max_seq_len有没有超过设定的300
	* （3）placeholder要和模型中的一致
