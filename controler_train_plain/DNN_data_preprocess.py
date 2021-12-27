import os
import numpy as np
import re

def write_input(f,l):
	for p in range(len(l)):
		tmp=l[p]
		tmp_round=tmp
		#f.write(str(tmp_round))
		f.write((',').join(str(i) for i in tmp_round))
		f.write('\n')
	return

f1=[]
f2=[]
x=[]
y=[]
for dirname, _, filenames in os.walk('data/input'):
    for filename in filenames:
        f1.append(os.path.join(dirname, filename))
        #print(os.path.join(dirname, filename))
for dirname, _, filenames in os.walk('data/output'):
    for filename in filenames:
        f2.append(os.path.join(dirname, filename))
        #print(os.path.join(dirname, filename))
f1=sorted(f1)
f2=sorted(f2)
prefix=[]
for i in range(11):
	if '4' in f1[i] and '4' in f2[i]:
		prefix.append(i)
print(prefix)
for ind in prefix:
	print(f1[ind],f2[ind])
	with open(f1[ind],"r") as filestream:
	    for line in filestream:
	        a=[]
	        currentline=line.split(',')
	        for j in range(34):
	            a.append(float(currentline[j]))
	        x.append(a)
	    print(np.array(x).shape)
	    x.pop()
	    print(np.array(x).shape)
	with open(f2[ind],"r") as filestream:
	    for line in filestream:
	        b=[]
	        currentline=line.split(',')
	        for j in range(12):
	            b.append(float(currentline[j]))
	        y.append(b)
	    print(np.array(y).shape)

#####################################need to be changed
f=open('data/processed/output/4_highstair_processed_545_output.txt','w')
f.truncate(0)
print(len(y))
#write_input(f,x)
write_input(f,y)
#print(np.array(x).shape,np.array(y).shape)