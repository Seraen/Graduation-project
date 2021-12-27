import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from DNN_model import DNN
#from DNN_controler_train import x_mean,x_std,y_mean,y_std,LEARNING_RATE,optimizer
from DNN_train_single import x_min,x_max,y_min,y_max,LEARNING_RATE,optimizer


#input_data=[-0.001106,0.005581,-0.00014,0.999984,0.005352,0.010539,0.031308,-0.057548,-0.255405,-0.058802,0.057226,-0.026536,0.494909,-0.367026,1.287766,-5.764668,-0.094418,1.42598,-5.921732,
            #-0.023836,0.399491,-0.180564,-0.006146,0.91004,-1.795579,-0.000597,0.918047,-1.862051,0.006645,0.907309,-1.838026,0.00309,0.886912,-1.775175]
#output_data=[-0.006104,0.910128,-1.794883,-0.00109,0.920935,-1.874648,0.00617,0.910468,-1.85086,0.002967,0.887808,-1.775744]

# use minmax normalize
f=open('crazy_process_of_plane_data.txt','w')
f.truncate(0)
def write_input(f,l):
	f.write((',').join(str(indx) for indx in l))
	f.write('\n')
	return

def get_dnn_output(input_dnn):
	input_data=[]
	output_dnn=[]
	for q in range(34):
		#tmp=(input_dnn[q]-x_mean[q])/x_std[q]
		tmp = (input_dnn[q] - x_min[q]) / (x_max[q] - x_min[q])
		input_data.append(tmp)
	input_data=torch.tensor(input_data).to(torch.float32)
	output_data=model(input_data)
	output_data=output_data.tolist()
	#print(output_data)
	#print(y_mean)
	#print(y_std)
	# output_dnn needs to be implemented to the quad
	for w in range(12):
		#tmp=output_data[w]*y_std[w]+y_mean[w]
		tmp = output_data[w] * (y_max[w] - y_min[w]) + y_min[w]
		output_dnn.append(tmp)
	return output_dnn

model_dir='model_single_test/plane_single_test/200_epoch.pth'
model=DNN()
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']

#count=0
with open('data/input/2_plane_input.txt', "r") as filestream:
	for line in filestream:
		a = []
		currentline = line.split(',')
		for j in range(34):
			a.append(float(currentline[j]))
		#a.append(np.sin((count % 50) / 50 * 2 * np.pi))
		#a.append(np.cos((count % 50) / 50 * 2 * np.pi))
		output_data=get_dnn_output(a)
		#count+=1
		#print(output_data)
		write_input(f,output_data)


'''
# use standardize
f=open('crazy_process_of_plane_data_standard.txt','w')
f.truncate(0)

def write_input(f,l):
	f.write((',').join(str(indx) for indx in l))
	f.write('\n')
	return

def get_dnn_output(input_dnn):
	input_data=[]
	output_dnn=[]
	for q in range(34):
		tmp=(input_dnn[q]-x_mean[q])/x_std[q]
		#tmp = (input_dnn[q] - x_min[q]) / (x_max[q] - x_min[q])
		input_data.append(tmp)
	input_data=torch.tensor(input_data).to(torch.float32)
	output_data=model(input_data)
	output_data=output_data.tolist()
	#print(output_data)
	#print(y_mean)
	#print(y_std)
	# output_dnn needs to be implemented to the quad
	for w in range(12):
		tmp=output_data[w]*y_std[w]+y_mean[w]
		#tmp = output_data[w] * (y_max[w] - y_min[w]) + y_min[w]
		output_dnn.append(tmp)
	return output_dnn

model_dir='model_single_test/plane_single_test/200_epoch.pth'
model=DNN()
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']

#count=0
with open('data/input/2_plane_input.txt', "r") as filestream:
	for line in filestream:
		a = []
		currentline = line.split(',')
		for j in range(34):
			a.append(float(currentline[j]))
		#a.append(np.sin((count % 50) / 50 * 2 * np.pi))
		#a.append(np.cos((count % 50) / 50 * 2 * np.pi))
		output_data=get_dnn_output(a)
		#count+=1
		#print(output_data)
		write_input(f,output_data)
'''