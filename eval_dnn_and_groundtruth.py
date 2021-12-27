import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DNN_model import DNN
import torchnet
import argparse
import logging
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,TensorDataset


f1='crazy_process_of_plane_data.txt'
f2='data/output/2_plane_output.txt'
x=[]
y=[]

with open(f1,"r") as filestream:
    for line in filestream:
        a=[]
        currentline=line.split(',')
        for j in range(12):
            a.append(float(currentline[j]))
        x.append(a)
    x.pop()
with open(f2,"r") as filestream:
    for line in filestream:
        b=[]
        currentline=line.split(',')
        for j in range(12):
            b.append(float(currentline[j]))
        y.append(b)
print(np.array(x).shape,np.array(y).shape)

x=np.array(x)
y=np.array(y)

def get_mse(x,y):
    if len(x)==len(y):
        return np.sum((x - y) ** 2)/len(x)
    else:
        return None


plot_dnn=[]
plot_controller=[]
#loss_function = nn.MSELoss()
for j in range(12):
    plot_dnn=[]
    plot_controller=[]
    for i in range(1000):
       tmp1=x[i+100][j]
       tmp2=y[i+100][j]
       plot_dnn.append(tmp1)
       plot_controller.append(tmp2)
    #loss=loss_function(x,y)
    #print(loss)
    # for i in range(len(x)):
    #     tmp=sum(x[i])
    #     total.append(tmp)
    plt.figure()
    plt.plot(plot_dnn,'blue')
    plt.plot(plot_controller,'red')
    plt.savefig('fig/'+str(j)+'.png')
    #plt.show()