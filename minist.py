#-*-coding: utf-8 -*-
'''
Function: test MINIST dataset by keras
Date:     2018.6.20
Author:   Eric.M
Email:    master2017@163.com
'''

import numpy as np
np.random.seed(1337)

#导入序列模型
from keras.models import Sequential

#导入Dense, Dropout, Activation
from keras.layers.core import Dense, Dropout, Activation

#从tensorflow载入数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train, y_train = mnist.train.images, mnist.train.labels
x_test, y_test = mnist.test.images, mnist.test.labels
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32')
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32')

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# keras中的mnist数据集已经被划分成了55,000个训练集，10,000个测试集的形式，按以下格式调用即可
# x_train原本是一个60000*28*28的三维向量，将其转换为60000*784的二维向量
# x_test原本是一个10000*28*28的三维向量，将其转换为10000*784的二维向量
#修改维度
x_train = x_train.reshape(55000,784)
x_test = x_test.reshape(10000,784)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# 将x_train, x_test的数据格式转为float32存储
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#归一化
x_train = x_train / 255
x_test = x_test / 255

# 打印出训练集和测试集的信息
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 建立顺序型模型
model = Sequential()

'''
模型需要知道输入数据的shape，
因此，Sequential的第一层需要接受一个关于输入数据shape的参数，
后面的各个层则可以自动推导出中间数据的shape，
因此不需要为每个层都指定这个参数
'''

# 输入层有784个神经元
# 第一个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 第二个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 输出层有10个神经元，激活函数为SoftMax，得到分类结果
model.add(Dense(10))
model.add(Activation('softmax'))

# 输出模型的整体信息
# 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size = 200,
                    epochs = 20,
                    verbose = 1,
                    validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])

