import pybullet as p
cin = p.connect(p.SHARED_MEMORY)
if (cin < 0):
    cin = p.connect(p.GUI)
objects = [p.loadURDF("bilibili/quadruped_model/a1/a1.urdf", -0.040416,-0.009217,0.406544,0.156267,-0.078018,-0.081282,0.981268)]
ob = objects[0]
jointPositions=[ 0.568760, 0.128918, -1.075653, 0.000000, -0.604565, 0.161106, -0.910103, 0.000000, 0.009261, -0.420788, -0.916197, 0.000000, -0.187291, 0.530370, -1.203363, 0.000000 ]
for jointIndex in range (p.getNumJoints(ob)):
	p.resetJointState(ob,jointIndex,jointPositions[jointIndex])

p.setGravity(0.000000,0.000000,-9.800000)
p.stepSimulation()
p.disconnect()
