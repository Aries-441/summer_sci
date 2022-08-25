from distutils.sysconfig import customize_compiler
import keras.optimizers.optimizer_v1
import xlrd3, xlwt,openpyxl
import os
import math
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import layers, models
from tabnanny import verbose
from keras import backend as K
from playsound import playsound
import winsound
from time import sleep

#全局变量
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
read_batch=2000

#定义神经网路
def build():
    model = models.Sequential()
    model.add(layers.Conv1D(32, 2, activation = "relu", padding='same', input_shape=(256,1)))
    model.add(layers.Conv1D(32, 2, activation = "relu", padding='same'))
    model.add(layers.MaxPooling1D(2,strides=2))#output 128
    model.add(layers.Conv1D(64, 2, activation = "relu", padding='same'))
    model.add(layers.Conv1D(64, 2, activation = "relu", padding='same'))
    model.add(layers.MaxPooling1D(2,strides=2))#output 64
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.27))#防止过拟合，需要调整
    model.add(layers.Dense(16))
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam',loss='mean_squared_error')
    return model

#加载原始数据
#处理数据后保存为新文件
def make_xlsx():
    wbn = xlrd3.open_workbook("C:\\Users\\15182\\Desktop\\summer_sci\\LIT101_Normal.xlsx")
    worksheet_normal = wbn.sheet_by_index(0)
    colvalues_normal = np.array(worksheet_normal.col_values(1,16000,496000), dtype='float32').reshape((-1,1))#舍弃前期不稳定数据
    Max_normal = max(worksheet_normal.col_values(1,16000,496000),key = abs)#寻找最大值
    print(Max_normal)
    for i in range(0,len(colvalues_normal)):
        colvalues_normal[i] = colvalues_normal[i] / Max_normal
        i+=1 
    data = pd.DataFrame(colvalues_normal)
    writer = pd.ExcelWriter('Traindata.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save() 
       
    wba = xlrd3.open_workbook(u"C:\\Users\\15182\\Desktop\\summer_sci\\LIT101_Attack.xlsx")
    worksheet_attack = wba.sheet_by_index(0)
    colvalues_attack = np.array(worksheet_attack.col_values(1,1,449000), dtype='float32').reshape((-1,1))#舍弃前期不稳定数据
    Max_attack = max(worksheet_attack.col_values(1,1,449000),key = abs)#寻找最大值
    print(Max_attack)
    for i in range(0,len(colvalues_attack)):
        colvalues_attack[i] = colvalues_attack[i] / Max_attack
        i+=1 
    data = pd.DataFrame(colvalues_attack)
    writer_1 = pd.ExcelWriter('Testdata.xlsx')  # 写入Excel文件
    data.to_excel(writer_1, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer_1.save()

#获取训练时train
def get_train_data(sheet, col):
    while 1:
        for k in range(0, 200000, read_batch):
            input_layour = np.array(sheet.col_values(col, k, k + read_batch + 255),  dtype='float32').reshape((-1, 1, 1))
            output_layour = np.array(sheet.col_values(col, k + 256, k + read_batch + 256),  dtype='float32').reshape((-1, 1))#xiugai
            input_layour = np.lib.stride_tricks.as_strided(input_layour, shape=(output_layour.shape[0], 256, 1),
                                                           strides=(input_layour.strides[0], input_layour.strides[0],
                                                                    input_layour.strides[0]))
            for j in range(0,read_batch,20):
                yield input_layour, output_layour
            del input_layour, output_layour

#寻找最大绝对偏差值，以及计算最大绝对偏差比率
def get_max():
    wb_train = xlrd3.open_workbook(u'C:\\Users\\15182\Desktop\\summer_sci\\Traindata.xlsx')
    worksheet_train = wb_train.sheet_by_index(0)
    max_abs_bias = 0
    max_abs_percentage = 0
    mean_abs_bias = 0
    mean_abs_percentage = 0
    mean_bias = 0
    standard_deviation = 0
    for i in range(1,400000,read_batch):
        input_layer=np.array(worksheet_train.col_values(1, i, i + read_batch + 255), dtype='float32').reshape((-1,1,1))
        ans_output=np.array(worksheet_train.col_values(1, i + 256, i + read_batch + 256)).reshape((-1, 1))#model.predict(input_layer, verbose=0)
        input_layer = np.lib.stride_tricks.as_strided(input_layer, shape=(ans_output.shape[0], 256, 1),
                                                      strides=(input_layer.strides[0], input_layer.strides[0],
                                                               input_layer.strides[0]))
        predicted_train = model.predict(input_layer, verbose=0)*18.3
        bias = np.array(predicted_train - ans_output,  dtype='float32').reshape((-1, 1))
        value = np.array((bias), dtype='float32').reshape((-1, 1)) 
        standard_deviation += np.sum(value**2) 
        mean_abs_bias += np.mean(abs(bias))
        mean_abs_percentage += np.mean(abs(bias) / ans_output)
        mean_bias += np.mean(bias)
        if np.max(abs(bias)) > max_abs_bias:
            max_abs_bias = np.max(abs(predicted_train - ans_output))            
        if np.max(abs(bias) / ans_output) > max_abs_percentage:
            max_abs_percentage = np.max(abs(bias) / ans_output)
    standard_deviation = math.pow((standard_deviation / 400000),0.5)
    print('standard_deviation: ',standard_deviation)    
    print('max_abs_bias: ',max_abs_bias)
    print('max_abs_percentage: ',max_abs_percentage)    
    print('mean_abs_bias: ',mean_abs_bias/(400000/2000))
    print('mean_abs_percentage: ',mean_abs_percentage/(400000/2000))
    print('mean_bias: ',mean_bias / (400000/2000))
    return mean_bias,standard_deviation   
    
#寻找攻击点位
#点位：
#4000
#116000
#360000
#388000
def begin_test():
    wb_train = xlrd3.open_workbook(u'C:\\Users\\15182\\Desktop\\summer_sci\\Testdata.xlsx')
    worksheet_train = wb_train.sheet_by_index(0)
    max_z = 0.08
    for i in range(0,399999-2000-257,2000):
        input_layer=np.array(worksheet_train.col_values(1, i, i + read_batch + 255), dtype='float32').reshape((-1,1,1))
        ans_output=np.array(worksheet_train.col_values(1, i + 256, i + read_batch + 256)).reshape((-1, 1))#model.predict(input_layer, verbose=0)
        input_layer = np.lib.stride_tricks.as_strided(input_layer, shape=(ans_output.shape[0], 256, 1),
                                                      strides=(input_layer.strides[0], input_layer.strides[0],
                                                               input_layer.strides[0]))
        predicted_train = model(input_layer)
        bias = abs(predicted_train - ans_output)
        num = 0
        for j in range(0,2000,20) :
            if bias[j] > max_z :
                for k in range(j,min(j+128,2000),1):
                    if bias[k] > max_z :
                        num += 1
                if num / min(128,(2000-j)) > 0.3:
                    print("Attack Detected ! : ",i+j,'  ',bias[j],'  ',num / min(128,(2000-j)))
                    num = 0
                    break
                else:
                    num =0

#调用处理数据函数
while (1==1):
    make_xlsx()
    break

#加载处理后的数据作为训练时test
wb = xlrd3.open_workbook(u'C:\\Users\\15182\\Desktop\\summer_sci\\Traindata.xlsx')
worksheet_test = wb.sheet_by_index(0)
test_in = np.array(worksheet_test.col_values(1, 450000, 450000 + read_batch + 255),
                   dtype='float32').reshape((-1, 1, 1))
test_out = np.array(worksheet_test.col_values(1, 450000 + 256, 450000 + read_batch + 256)).reshape((-1, 1))
test_in = np.lib.stride_tricks.as_strided(test_in, shape=(test_out.shape[0], 256, 1),
                                              strides=(test_in.strides[0], test_in.strides[0],
                                              test_in.strides[0]))


#训练过程，可以跳过，直接调用训练好的模型，开始测试
while (1==0):
    history = build().fit(get_train_data(worksheet_test,1), epochs=4, steps_per_epoch=2000, batch_size=100,
                validation_data=(test_in, test_out), validation_batch_size=100)#5~10次便可以实现收敛，次数过多容易导致过拟合
    build().save('1D_CNN_ICS.h5')
    break

#调用已经训练完毕的模型
model=tf.keras.models.load_model('1D_CNN_ICS.h5')
print('\n',"The model test_id:006 has been successfully called! ",'\n')
for i in range(1,4):
        winsound.Beep(2000,300)
        winsound.Beep(500,300)
        sleep(1)
model.summary()


while (1 ==0):
    #加载处理后的数据作为 验证数据
    wb = xlrd3.open_workbook(u'C:\\Users\\15182\\Desktop\\summer_sci\\Traindata.xlsx')
    worksheet_test = wb.sheet_by_index(0)
    #截取验证集合用以绘图
    input_layour = np.array(worksheet_test.col_values(1, 450000, 450000 + read_batch + 255), dtype='float32').reshape((-1, 1, 1))
    output_layour = np.array(worksheet_test.col_values(1,  450000 + 256, 450000 + read_batch + 256 ), dtype='float32').reshape((-1, 1))
    input_layour = np.lib.stride_tricks.as_strided(input_layour, shape=(output_layour.shape[0], 256, 1),
                                                                strides=(input_layour.strides[0], 
                                                                        input_layour.strides[0],
                                                                input_layour.strides[0]))
    predicted = model.predict(input_layour,batch_size = 100)
    #绘制偏差图 test_bias
    x=np.linspace(0.5,1.1,100)
    y=np.linspace(0.5,1.1,100)
    plt.plot(output_layour,predicted,color='green',linewidth=1.0,linestyle='--',label='line')
    plt.plot(x,y,color='red',linewidth=1.0,linestyle='--',label='line')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(["传感器读数预测值","x = y"])
    plt.title("预测值与真实值de的偏差")
    plt.xlabel('传感器特征值真实值')
    plt.ylabel('传感器特征值预测值')
    plt.savefig('test_bias.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
    #绘制时间序列图 test_time
    bias = abs(output_layour - predicted)*20
    percentage = (bias/output_layour)/20
    x=np.linspace(1,2000,2000)
    plt.plot(x,predicted,color='green',linewidth=1.0,linestyle='--',label='line')
    plt.plot(x,output_layour,color='red',linewidth=1.0,linestyle='--',label='line')
    plt.plot(x,bias,color='black',linewidth=1.0,linestyle='--',label='line')
    plt.plot(x,percentage,color='yellow',linewidth=1.0,linestyle='--',label='line')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend(["传感器读数预测值","传感器读数真实值","绝对误差(*20)","绝对误差比率"])
    plt.title("预测值与真实值随时间序列的变化")
    plt.xlabel('时间序列')
    plt.ylabel('特征值')
    plt.savefig('test_time.png', dpi=200, bbox_inches='tight', transparent=False)
    plt.show()
    break




#调用，查看在normal_dataset中最大值
#mean_bias,standard_deviation = get_max()
#begin_test()


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs, loss, 'bo',label = 'Training loss')
plt.plot(epochs, val_loss, 'b',label = 'Validation loss')
plt.legend(["loss","val_loss"])
plt.title('Training and validation loss')
plt.savefig('Training_and_validation_loss', dpi=200, bbox_inches='tight', transparent=False)
plt.show()




                
                
        


