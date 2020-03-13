
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

f=open('Challenge_Data/train.txt')
datatrain=f.read()
datatrain=datatrain.split()
datatrain=np.array(datatrain,dtype=np.float32)
datatrain=datatrain.reshape(-1,26)
datatrain=datatrain.transpose()
v=datatrain.shape[0]
tp=datatrain.shape[1]

listdata=[]
symbol=0
for i in range(tp-1):
    if datatrain[0,i+1]-datatrain[0,i]==1:
        unit=datatrain[:,symbol:(i+1)]
        symbol=i+1
        listdata.append(unit)
listdata.append(datatrain[:,symbol:-1])

#r=25
#col=listtrain[r].shape[1]
#for j in range(26):
#    t=np.linspace(0,col,col,dtype=np.float32)
#    plt.figure()
#    plt.plot(t,listtrain[r][j])
    
num=len(listdata)
labeldata=[]
list21=[]
for i in range(num):
    vi=listdata[i].shape[1]
    label=np.zeros([1,vi])
    list21.append(listdata[i][3:])
    for j in range(vi):
        if vi-j>130:
            label[0,j]=vi-j
        else:
            label[0,j]=vi-j
    labeldata.append(label)

listdata=list21

for i in range(num):
    for j in range(listdata[i].shape[0]):
        listdata[i][j]=(listdata[i][j]-np.min(listdata[i][j]))/(np.max(listdata[i][j])-np.min(listdata[i][j]))
    labeldata[i]=(0.8*labeldata[i]-np.min(labeldata[i]))/(np.max(labeldata[i])-np.min(labeldata[i]))
    listdata[i]=listdata[i][2:]
    
PCA_len=25
listeigen=[]
for i in range(num):
    thi=listdata[i]
    leng=thi.shape[1]
    eigen=np.zeros([thi.shape[0],leng-PCA_len])
    for j in range(leng-PCA_len):
        ma=np.matrix(thi[:,j:j+PCA_len])
        ma_cov=ma*np.transpose(ma)
        eigenvalue,eigenvector=np.linalg.eig(ma_cov)
        eigen[:,j]=eigenvalue
    listeigen.append(eigen)
    



lins=np.linspace(1,listeigen[2].shape[1],listeigen[2].shape[1],dtype=np.float32)
plt.plot(lins,listeigen[2][2])


for i in range(21):
    t=81
    lins=np.linspace(1,listeigen[t].shape[1],listeigen[t].shape[1],dtype=np.float32)
    plt.plot(lins,listeigen[t][i])

for i in range(20):
    lins=np.linspace(1,350,350,dtype=np.float32)
    zz=np.zeros(lins.shape)
    lej=listeigen[i].shape[1]
    zz[350-lej:350]=listeigen[i][2]
    plt.plot(lins,zz)

#    
#    
#
#''' 218 data is splited as 180 training data and 38 testing data'''
#split=180
#
#listtrain=listdata[:180]
#listtest=listdata[180:]
#labeltrain=labeldata[:180]
#labeltest=labeldata[180:]
#    