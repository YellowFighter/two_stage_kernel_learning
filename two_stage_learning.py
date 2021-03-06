# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:49:47 2018

@author: Lenovo0705
此程序用来仿真two-stage sampled learning theory on distributions文章的第二个例子aerosol prediction
数据集来源于MISR1数据集"E:/two_stage/phase_learn_master/data/MISR1.mat"
文章分别对加噪和不加噪分别进行了研究
最新的方法在Testing and Learning on Distributions with Symmetric Noise Invariance中取得了很好的效果
程序参考Phase_Neural_Network_Aerosol.ipynb
本程序意在使用kernel embedding的方法测试
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ite
from ite.cost.x_kernel import Kernel
from sklearn.metrics import mean_squared_error
from phase_learn_master import aux_fct
from phase_learn_master import phase_fourier_dr_nn
import warnings
warnings.filterwarnings('ignore')

path = 'E:/two_stage/phase_learn_master/data/' #Change path 
# Load Dataset into features and labels 产生数据所用的方法是原程序中的，此处仅为方便故调用
misr_data_x, misr_data_y = aux_fct.load_data(path, random = True)
variance = np.var(np.concatenate(misr_data_x), axis = 0) # For calculating signal to noise ratio

X_train = misr_data_x[:480]
y_train = misr_data_y[:480]
X_val = misr_data_x[480:640]
y_val = misr_data_y[480:640]
X_test = misr_data_x[640:]
y_test = misr_data_y[640:]
l = X_train.shape[0]
m = X_val.shape[0]

#每个sample包含16维特征，其中包括12维的反射比数值和4个测量角，每个bag包含100个sample，共800个bag  其中480组用来训练，即计算核相关K矩阵为640*640，
#验证集160组数据用来调参，主要是用来在给定的范围内选择最优的theta和lamda，测试集160组，用于验证方法的可行性
#在数据集load的过程中已经包含了数据集的打乱，通过aux_fct.load_data(path, random = True，seed=1)改变不同的seed值来重复不同划分数据集以取平均
#来使结果更具说服力

class kernel_learning:
        
    def __init__(self,theta,lamda):
        self.theta = theta
        self.lamda = lamda
        
    def com_kernel(self):
        k2 = Kernel({'name': 'RBF','sigma': self.theta})
        self.co = ite.cost.BKExpected(kernel=k2)     #此处参考ite的用法
        return self.co

    def com_k(self,t,dataset):  #当为训练时dataset使用val集，当为验证效果时用test集,l为训练集样本个数,m为验证集样本数,t为dataset集中的每一个
        k = np.zeros((l,1))
        for i in range(l):
            k[i] = self.co.estimation(X_train[i],dataset[t])          #给定一个新的样本，他与480组训练样本的每一个instance都要进行kernel的运算
        return k       
    def com_y(self,dataset):   
        K = np.zeros((l,l))
        for i in range(l):
            for j in range(l):
                K[i][j] = self.co.estimation(X_train[i],X_train[j])      #计算K矩阵
        y_pred = np.zeros(m)                        
        for i in range(m):       
            y_pred[i] = np.matmul(np.matmul(y_train.reshape(1,-1),np.linalg.inv((K+l*self.lamda*np.eye(l)))),\
                  self.com_k(i,dataset))#用于计算经过kernel embedding之后的值
        return y_pred

def search_best_para():
        minerror = 100000
        for theta in np.logspace(-15,10,num=26,base=2):
            for lamda in np.logspace(-65,-3,num=63,base=2):   #两个参数的搜索范围，为以2为底的指数，注意写法
                a = kernel_learning(theta,lamda)
                b = a.com_kernel()
                y_pred = a.com_y(X_val)
                sumerror = mean_squared_error(y_pred,y_val)
                if sumerror < minerror:
                    minerror = sumerror
                    besttheta = theta
                    bestlamda = lamda
        f = plt.figure(1)
        plt.scatter(y_pred,y_val)
        plt.show()
        print (besttheta)
        print (bestlamda)

def plot_bestpara_val():  #用于画出验证集
    theta = 128
    lamda = 6.103515625e-05 
    a = kernel_learning(theta,lamda)
    b = a.com_kernel()
    y_pred = a.com_y(X_val)
    f = plt.figure(2)
    plt.scatter(y_pred,y_val)
    plt.title("Relation on validation dataset")
    plt.ylabel("Entroy")
    plt.xlabel("Embedding of validation dataset")
    plt.show()
    
def testresult():  #用于在测试集上验证结果，与上面函数的差别主要在于k_small的用的是X_test而搜索参数用的是X_val
    theta = 128
    lamda = 6.103515625e-05 
    a = kernel_learning(theta,lamda)
    b = a.com_kernel()
    y_pred = a.com_y(X_test)
    f = plt.figure(3)
    plt.title("Relation on test dataset")
    plt.ylabel("True AOD")
    plt.xlabel("Prediction of AOD")
    plt.xlim(-0.01,1.25)
    plt.ylim(-0.01,1.25)
    mse = mean_squared_error(y_pred,y_test)
    rmse = np.sqrt(mse)
    print ("RMSE is ",rmse)
    plt.scatter(y_pred,y_test)
    plt.show()

#search_best_para()    #搜索参数时间太长，本程序中所使用参数为搜索得到的最优参数
#plot_bestpara_val()
testresult()
#需要执行哪个程序将注释去掉即可