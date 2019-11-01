from Network import *
from Read_file import *
from sklearn.model_selection import train_test_split
from datetime import datetime
from matplotlib import pyplot as plt

INPUT_NODE = 512
OUTPUT_NODE = 512

EPOCHS = 1          # 训练轮数
BATCHS = 700       # 一轮训练次数
test_times = 1e2    # 测试次数
BATCH_SIZE = 200      # 一批数量

LEARNING_RATE_BASE = 0.001  # 学习速率
LEARNING_RATE_BASE_Vary_times = BATCHS  # 学习速率多少次变化一次

EPOCHS_num = BATCHS     # 一次epoch的数量
FileTotal_num = 3e5     # 一个CSV文件的总大小
TEST_Factor = 0.2       # 测试因子
Check_Factor = 0.2      # 校验因子
TEST_SIZE = FileTotal_num*TEST_Factor    # 测试数量
Check_SIZE = FileTotal_num*Check_Factor  # 校验数量
regularizer_rate = 0.0001       # 正则化系数

STAIRCASE = True
tic = datetime.now()    # 计时函数

#---------------------------------------------------------------------------------------------
#                        网络配置
#---------------------------------------------------------------------------------------------

x = tf.placeholder(tf.float32, [INPUT_NODE, BATCH_SIZE], name='x_input')    # 输入

y_ = tf.placeholder(tf.float32, [OUTPUT_NODE, BATCH_SIZE], name='y_input')  # 标签

y = New_Net_Dnn(x, INPUT_NODE, OUTPUT_NODE,BATCH_SIZE)  # 网络输出

# 损失函数
# loss = tf.reduce_mean( tf.square(tf.norm((y_ - y),axis=0)) / (tf.square(tf.norm(y_,axis=0))))
# loss = tf.div(tf.reduce_sum(tf.square(y_ - y)),tf.reduce_sum(tf.square(y_)))
# loss = tf.reduce_mean(tf.pow(y_ - y, 2))
loss=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,-1.0,1.0)))
# 定义当前迭代轮数的变量
global_step = tf.get_variable('global_step',dtype=tf.int32, initializer=0,trainable=False)  # 不可训练
# 指数衰减学习速率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,LEARNING_RATE_BASE_Vary_times,
                                           0.96,
                                           staircase=STAIRCASE)
# 定义优化函数
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
#-----------------------------------------------------------------------------------------------
#                             开始训练
#-----------------------------------------------------------------------------------------------

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:

    train_list_all = []
    test_list_all = []
    verify_list_all = []
    for i in range(EPOCHS):
        train_loss_list = []
        test_loss_list = []
        verify_loss_list = []
        X,Y = read_data()
        # 分离训练集与测试集以及校验集
        (X_train, X_TandC, Y_train, Y_TandC) = train_test_split(X.T, Y.T, test_size=int(TEST_SIZE+Check_SIZE))
        (X_test, X_check, Y_test, Y_check) = train_test_split(X_TandC, Y_TandC, test_size=int(Check_SIZE))
        (X_train, X_test,X_check, Y_train, Y_test,Y_check) = (X_train.T, X_test.T,X_check.T, Y_train.T, Y_test.T,Y_check.T)  # 转置
        tf.global_variables_initializer().run()  # 初始化
        #---------------训练开始-----------------------------------------------------------
        for j in range(BATCHS):
            Output_X, Output_Y = data_choose(X_train, Y_train, BATCH_SIZE)
            # X_after_nor = (Output_X - np.mean(Output_X)) / np.std(Output_X)
            X_after_nor = Output_X
            sess.run(train_step, feed_dict={x: X_after_nor, y_: Output_Y})
            train_loss = sess.run(loss, feed_dict={x: X_after_nor, y_: Output_Y})
            train_loss_list.append(train_loss)
            print('训练了%d次,训练集损失%.12f' % (j, train_loss))
            #-----------校验开始-----------------------------------------------------------
            if j%10 == 0:
                Output_check_X,Output_check_Y = data_choose(X_check, Y_check, BATCH_SIZE)
                # X_after_nor_CHECK = (Output_check_X - np.mean(Output_check_X)) / np.std(Output_check_X)
                X_after_nor_CHECK = Output_check_X
                verify_loss = sess.run(loss, feed_dict={x: X_after_nor_CHECK, y_: Output_check_Y})
                verify_loss_list.append(verify_loss)
        #---------------测试开始-----------------------------------------------------------
        for k in range(int(test_times)):
            Output_test_X, Output_test_Y = data_choose(X_test, Y_test, BATCH_SIZE)
            # X_after_nor_TEST = (Output_test_X - np.mean(Output_test_X)) / np.std(Output_test_X)
            X_after_nor_TEST = Output_test_X
            test_loss = sess.run(loss, feed_dict={x: X_after_nor_TEST, y_: Output_test_Y})
            test_loss_list.append(test_loss)

        train_list_all.append(train_loss_list)
        test_list_all.append(test_loss_list)
        verify_list_all.append(verify_loss_list)

    toc = datetime.now()
    # 训练集损失图
    pd.DataFrame(train_list_all, index=['loss_1']).T.plot()
    plt.title('Train loss')
    plt.show()
    # 校验集损失图
    pd.DataFrame(verify_list_all, index=['loss_1']).T.plot()
    plt.title('Check loss')
    plt.show()
    # 训练信噪比下的图
    pd.DataFrame(test_list_all, index=['loss_1']).T.plot()
    plt.title('Test loss')
    plt.show()











