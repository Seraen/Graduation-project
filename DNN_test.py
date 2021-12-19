import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from DNN_model import DNN
#from DNN_controler_train import x_mean,x_std,y_mean,y_std,LEARNING_RATE,optimizer
from DNN_train_single import x_min,x_max,y_min,y_max,LEARNING_RATE,optimizer
from bilibili.build.robot_controller import robot_controller
import pybullet as p
import time

spot_init_new_pos = [-0.1, 0.8, -1.6,
                0.1, 0.8, -1.6,
                -0.1, 0.8, -1.6,
                0.1, 0.8, -1.6]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class Dog:
p.connect(p.GUI)
plane = p.loadURDF("urdf/plane.urdf")
p.setGravity(0,0,-9.8)
p.setTimeStep(1./500)
#p.setDefaultContactERP(0)
#urdfFlags = p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
urdfFlags = p.URDF_USE_SELF_COLLISION
quadruped = p.loadURDF("urdf/a1/urdf/a1.urdf",[0,0,0.4],[0,0,0,1], flags = urdfFlags,useFixedBase=False)
p.resetBasePositionAndOrientation(plane, [0, 0, 0], [0, 0, 0, 1])

#enable collision between lower legs
for j in range (p.getNumJoints(quadruped)):
        print(p.getJointInfo(quadruped,j))

lower_legs = [2,5,8,11]
for l0 in lower_legs:
    for l1 in lower_legs:
        if (l1>l0):
            enableCollision = 1
            print("collision for pair",l0,l1, p.getJointInfo(quadruped,l0)[12],p.getJointInfo(quadruped,l1)[12], "enabled=",enableCollision)
            p.setCollisionFilterPair(quadruped, quadruped, 2,5,enableCollision)

jointIds=[]
paramIds=[]

maxForceId = p.addUserDebugParameter("maxForce",0,100,20)

for j in range (p.getNumJoints(quadruped)):
    p.changeDynamics(quadruped,j,linearDamping=0, angularDamping=0)
    info = p.getJointInfo(quadruped,j)
    # print(info)
    jointName = info[1]
    jointType = info[2]
    if (jointType==p.JOINT_PRISMATIC or jointType==p.JOINT_REVOLUTE):
        jointIds.append(j)

# print(jointIds)

p.getCameraImage(480,320)
p.setRealTimeSimulation(0)

joints=[]

# load the model
model_dir='model_single_test/slope_single_test/200_epoch.pth'
model=DNN()
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']
# input_dnn and output_dnn intrinsic input and output
# input_data and output_data are processed
#input_dnn=[0.0,0.0,0.0,1.0,-0.000547,-0.005574,0.081215,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9,-1.8,0.0,0.9,-1.8,0.0,0.9,-1.8,0.0,0.9,-1.8]
#print(x_mean)
#print(x_std)

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

# orientation, base linear velocity, base angular velocity, joint velocity, 当前的joint position
def read_robot_state_input(quad):
	robot_state_input=[]
	robot_joints_vel=[]
	robot_joints_pos=[]
	_,orientation=p.getBasePositionAndOrientation(quad)
	#robot_state_input.append(i for i in orientation)
	robot_state_input+=orientation
	vel=p.getBaseVelocity(quad)
	#robot_state_input.append(i for i in vel[0])
	robot_state_input+=vel[0]
	#robot_state_input.append(i for i in vel[1])
	robot_state_input+=vel[1]
	for e in range(12):
		pres_pos=p.getJointState(quad,jointIds[e])
		robot_joints_pos.append(pres_pos[0])
		robot_joints_vel.append(pres_pos[1])
	#robot_state_input.append(i for i in robot_joints_vel)
	robot_state_input+=robot_joints_vel
	#robot_state_input.append(i for i in robot_joints_pos)
	robot_state_input+=robot_joints_pos
	return robot_state_input


output_dnn=[-0.006117,0.910351,-1.794465,-0.001334,0.924109,-1.888115,0.005445,0.913908,-1.864504,0.002785,0.888801,-1.77652]
robot_runner = robot_controller(1)
cnt=0
while (1):
	cnt+=1
	print(cnt)
	if cnt<1500:
		with open("mocap.txt", "r") as filestream:
			for line in filestream:
				cnt+=1
				if cnt>1500:
					break
				maxForce = p.readUserDebugParameter(maxForceId)
				currentline = line.split(",")
				# frame = currentline[0]
				# t = currentline[1]
				joints = currentline[2:14]
				for j in range(12):
					targetPos = float(joints[j])
					p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)
				input_dnn=read_robot_state_input(quadruped)
				p.stepSimulation()
				for lower_leg in lower_legs:
					# print("points for ", quadruped, " link: ", lower_leg)
					pts = p.getContactPoints(quadruped, -1, lower_leg)
				# print("num points=",len(pts))
				# for pt in pts:
				#    print(pt[9])
				time.sleep(1. / 500.)

	else:
		maxForce = p.readUserDebugParameter(maxForceId)
		for j_ind in range(12):
			targetPos = output_dnn[j_ind]
			p.setJointMotorControl2(quadruped, jointIds[j_ind], p.POSITION_CONTROL, targetPos, force=maxForce)
		p.stepSimulation()
		input_dnn = read_robot_state_input(quadruped)
		output_dnn = get_dnn_output(input_dnn)
		for lower_leg in lower_legs:
			# print("points for ", quadruped, " link: ", lower_leg)
			pts = p.getContactPoints(quadruped, -1, lower_leg)
		time.sleep(1. / 500.)

'''
while(1):
    with open("mocap.txt","r") as filestream:
        for line in filestream:
            maxForce = p.readUserDebugParameter(maxForceId)
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints=currentline[2:14]
            for j in range (12):
                targetPos = float(joints[j])
                p.setJointMotorControl2(quadruped, jointIds[j], p.POSITION_CONTROL, targetPos, force=maxForce)

            p.stepSimulation()
            for lower_leg in lower_legs:
                #print("points for ", quadruped, " link: ", lower_leg)
                pts = p.getContactPoints(quadruped,-1, lower_leg)
                #print("num points=",len(pts))
                #for pt in pts:
                #    print(pt[9])
            time.sleep(1./500.)
'''