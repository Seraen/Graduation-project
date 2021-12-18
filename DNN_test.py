import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from DNN_model import DNN
from DNN_controler_train import x_mean,x_std,y_mean,y_std,LEARNING_RATE,optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir='300_epoch.pth'
model=DNN()
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']
input_dnn=[-0.000119, 0.001289, 0.006996, 0.999975, 0.001833, -0.000561, 0.037586, -0.110096, 0.000179, -0.038858, 0.162871, -0.133677, 0.243046, 0.120607, -0.077007, 0.219075, 0.115132, -0.156494, 0.289351, 0.071821, -0.060062, 0.182478, -0.143549, 0.751118, -1.748382, 0.162994, 0.768374, -1.737127, -0.165435, 0.73994, -1.724106, 0.138618, 0.775632, -1.75075]
#print(x_mean)
#print(x_std)
input_data=[]
output_dnn=[]
for q in range(34):
	tmp=(input_dnn[q]-x_mean[q])/x_std[q]
	input_data.append(tmp)
input_data=torch.tensor(input_data).to(torch.float32)
output_data=model(input_data)
output_data=output_data.tolist()
#print(output_data)
#print(y_mean)
#print(y_std)
# output_dnn needs to be implemented to the quad
for p in range(12):
	tmp=output_data[p]*y_std[p]+y_mean[p]
	output_dnn.append(tmp)
print(output_dnn)
