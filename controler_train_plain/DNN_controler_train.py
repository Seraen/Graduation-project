import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from DNN_model import DNN
import torchnet
import argparse
import logging
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader,TensorDataset

batch_size=64
epochs=200
t_loss=0
v_loss=0
WORKERS=0
LEARNING_RATE=0.001
f1='data/input/2_plane_input.txt'
f2='data/output/2_plane_output.txt'
x=[]
y=[]
# read data from files
# for dirname, _, filenames in os.walk('data/input'):
#     for filename in filenames:
#         f1.append(os.path.join(dirname, filename))
#         #print(os.path.join(dirname, filename))
# for dirname, _, filenames in os.walk('data/output'):
#     for filename in filenames:
#         f2.append(os.path.join(dirname, filename))
#         #print(os.path.join(dirname, filename))
# f1=sorted(f1)
# f2=sorted(f2)
with open(f1,"r") as filestream:
    for line in filestream:
        a=[]
        currentline=line.split(',')
        for j in range(34):
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


# train-test split
x_train_pri, x_test_pri, y_train_pri, y_test_pri = train_test_split(x, y, test_size = 0.2)
X_train=x_train_pri
Y_train=y_train_pri
X_test=x_test_pri
Y_test=y_test_pri
'''
# data standalization
scaler = StandardScaler()
x_mean=np.mean(x_train_pri,axis=0)
x_std=np.std(x_train_pri,axis=0)
y_mean=np.mean(y_train_pri,axis=0)
y_std=np.std(y_train_pri,axis=0)
#print(x_mean[0],x_std[0],y_mean[0],y_std[0])
X_train = scaler.fit_transform(x_train_pri)
X_test=scaler.transform(x_test_pri)
#print(x_test_pri[0][1])
#print((x_test_pri[0][1]-x_mean[1])/x_std[1])
#print(X_test[0][1])
Y_train = scaler.fit_transform(y_train_pri)
Y_test=scaler.transform(y_test_pri)
'''

# transform to tensor
x_train=torch.tensor(X_train,dtype=torch.float32)
x_test=torch.tensor(X_test,dtype=torch.float32)
y_train=torch.tensor(Y_train,dtype=torch.float32)
y_test=torch.tensor(Y_test,dtype=torch.float32)
train_data=TensorDataset(x_train,y_train)
test_data=TensorDataset(x_test,y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=WORKERS)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=WORKERS)
#device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DNN model

# model and loss function
model = DNN()
loss_function = nn.MSELoss()
if device == 'cuda':
    model.cuda()
    loss_function.cuda()

#train
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_loss=[]
def train(epoch):
    global t_loss
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        if device=='cuda':
            data, target = data.float().cuda(), target.float().cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 and batch_idx>0:
            print('[Epoch %2d, batch %3d] training loss: %.4f' %
                (epoch, batch_idx, loss.data))
    t_loss=loss
    train_loss.append(loss)

#test
import torchnet as tnt
import math

test_loss=[]
def test():
    global v_loss
    model.eval()
    loss=0
    #test_loss = tnt.meter.AverageValueMeter()
    #top1 = tnt.meter.ClassErrorMeter()
    #correct=0
    with torch.no_grad():
        for data, target in test_loader:
            if device=='cuda':
                data, target = data.cuda(), target.cuda()
            #data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss += loss_function(output, target)
            #print(test_loss)
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            #top1.add(output.data, target.data)
            #test_loss.add(loss.data)
        #test_loss /= len(test_loader.dataset)
        loss/=math.ceil(len(test_loader.dataset)/batch_size)
    print('[Epoch %2d] Average test loss: %.6f\n'#, accuracy: %.2f%%\n'
        %(epoch, loss))#, top1.value()[0]))
    v_loss=loss
    test_loss.append(loss)
    '''
    print('[Epoch %2d] Average test loss: %.3f%%\n'#, accuracy: %.2f%%\n'
        %(epoch, test_loss.value()[0]))#, top1.value()[0]))
    '''

# train logger
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# plot loss image
def printfigure():
    plt.figure()
    plt.plot([x+1 for x in range(epochs)], test_loss, color='black', linestyle='-',label='validation loss')
    plt.plot([x+1 for x in range(epochs)], train_loss, color='red', linestyle='-',label='train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('judge overfitting')
    plt.show()

# train,test,save the model,save the log
logger = get_logger('model/plane_standard/exp1.log')
model_dir='model/plane_standard/'
if __name__=="__main__":
    logger.info('start training!')
    for epoch in range(1, epochs + 1):
        train(epoch)
        test()
        logger.info('Epoch:[{}/{}]\t train_loss={:.6f}\t test_loss={:.6f}'.format(epoch, epochs, t_loss, v_loss))
        state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state, model_dir+str(epoch)+'_epoch.pth')
    logger.info('finish training!')
    printfigure()

model_dir='model/plane_standard/200_epoch.pth'
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']
#test()