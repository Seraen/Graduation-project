import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from DNN_model import DNN
#from DNN_controler_train import x_mean,x_std,y_mean,y_std,LEARNING_RATE,optimizer
from DNN_train_single import x_min,x_max,y_min,y_max,LEARNING_RATE,optimizer
import pybullet as p
import time

model_dir='model_single_test/plain_single_test/200_epoch.pth'
model=DNN()
print(model)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']

def get_dnn_output(input_dnn):
	input_data=[]
	output_dnn=[]
	for q in range(34):
		tmp=(input_dnn[q]-x_min[q])/(x_max[q]-x_min[q])
		# tmp=(input_dnn[q]-x_mean[q])/x_std[q]
		input_data.append(tmp)
	input_data=torch.tensor(input_data).to(torch.float32)
	output_data=model(input_data)
	output_data=output_data.tolist()
	#print(output_data)
	#print(y_mean)
	#print(y_std)
	# output_dnn needs to be implemented to the quad
	for w in range(12):
		tmp=output_data[w]*(y_max[w]-y_min[w])+y_min[w]
		# tmp=output_data[w]*y_std[w]+y_mean[w]
		output_dnn.append(tmp)
	return output_dnn

#input=[-0.001106,0.005581,-0.00014,0.999984,0.005352,0.010539,0.031308,-0.057548,-0.255405,-0.058802,0.057226,-0.026536,0.494909,-0.367026,1.287766,-5.764668,-0.094418,1.42598,-5.921732,-0.023836,0.399491,-0.180564,-0.006146,0.91004,-1.795579,-0.000597,0.918047,-1.862051,0.006645,0.907309,-1.838026,0.00309,0.886912,-1.775175]
input =[-0.008578854072528822, 0.007945193401630427, -0.021005062178927367, 0.9997109905004908, 0.3411775342907737, -0.002037397604459803, -0.23208785114942038, -0.45359522768368643, 0.07868340208347838, 0.31223007645585577, 0.23699741443469902, -0.3814551187366568, 1.6760352212365486, -0.3182915781175762, 1.5337223985101691, -2.0281563015126, 0.26200021651786376, 1.501517217998099, -2.5757829728503974, 0.1354658700304885, -1.231230321806089, 1.417411843195103, 0.08821452856409165, 0.8637515268434769, -1.9040459940805863, -0.07917043277381637, 1.0794145559627821, -1.8564324287406166, 0.012080815155391991, 1.0199868257974283, -1.7722246095248078, -0.0013028542230364547, 0.8153289534396835, -1.798149232475382]
output=get_dnn_output(input)
print(output)
# -0.006104,0.910128,-1.794883,-0.00109,0.920935,-1.874648,
# 0.00617,0.910468,-1.85086,0.002967,0.887808,-1.775744
#joints ['0.09248059944581985', '0.8568853347062171', '-1.8738769798871875', '-0.0849038422179222', '1.1070215503304004', '-1.8929400022339822', '0.016790213889479655', '1.0470131285133362', '-1.8185891235563756', '0.001133277894496923', '0.7931660438861847', '-1.7726365662083028\n']
#read [0.08821452856409165, 0.8637515268434769, -1.9040459940805863, -0.07917043277381637, 1.0794145559627821, -1.8564324287406166, 0.012080815155391991, 1.0199868257974283, -1.7722246095248078, -0.0013028542230364547, 0.8153289534396835, -1.798149232475382]
#input_dnn [-0.008578854072528822, 0.007945193401630427, -0.021005062178927367, 0.9997109905004908, 0.3411775342907737, -0.002037397604459803, -0.23208785114942038, -0.45359522768368643, 0.07868340208347838, 0.31223007645585577, 0.23699741443469902, -0.3814551187366568, 1.6760352212365486, -0.3182915781175762, 1.5337223985101691, -2.0281563015126, 0.26200021651786376, 1.501517217998099, -2.5757829728503974, 0.1354658700304885, -1.231230321806089, 1.417411843195103, 0.08821452856409165, 0.8637515268434769, -1.9040459940805863, -0.07917043277381637, 1.0794145559627821, -1.8564324287406166, 0.012080815155391991, 1.0199868257974283, -1.7722246095248078, -0.0013028542230364547, 0.8153289534396835, -1.798149232475382]
#input_joints [0.08821452856409165, 0.8637515268434769, -1.9040459940805863, -0.07917043277381637, 1.0794145559627821, -1.8564324287406166, 0.012080815155391991, 1.0199868257974283, -1.7722246095248078, -0.0013028542230364547, 0.8153289534396835, -1.798149232475382]
#output_dnn [0.06915484181106091, 0.9036518287235499, -1.9410328059620559, -0.05396501233100892, 0.9918699209858179, -1.9589759989391267, 0.0036529438628554317, 0.9472555308970213, -1.834589219007373, -0.007419471182256943, 0.8234197338906228, -1.8090230668900014]
#robot_joints_pos [0.0863085635234438, 0.8677415517421235, -1.9077448158691535, -0.07664976666827186, 1.0761783037194732, -1.8666868010902953, 0.011238235356594024, 1.0161032410957298, -1.7784610533049545, -0.0019146634594900892, 0.8161384207553009, -1.7996191659296807]