# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:50:01 2019

@author: A1881
"""


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
for i in range(num):
    vi=listdata[i].shape[1]
    label=np.zeros([1,vi])
    for j in range(vi):
        if vi-j>130:
            label[0,j]=130
        else:
            label[0,j]=vi-j
    labeldata.append(label)

for i in range(num):
    for j in range(listdata[i].shape[0]):
        listdata[i][j]=(listdata[i][j]-np.min(listdata[i][j]))/(np.max(listdata[i][j])-np.min(listdata[i][j]))
    labeldata[i]=(0.8*labeldata[i]-np.min(labeldata[i]))/(np.max(labeldata[i])-np.min(labeldata[i]))
    listdata[i]=listdata[i][2:]
''' 218 data is splited as 180 training data and 38 testing data'''
split=180

listtrain=listdata[:180]
listtest=listdata[180:]
labeltrain=labeldata[:180]
labeltest=labeldata[180:]
    
'''RNN'''

TIME_STEP = 50   #  rnn time step
INPUT_SIZE = 24      # rnn input size
OUT_SIZE = 1
CELL_SIZE = 8       # rnn cell size
LR = 0.002        # learning rate
layer_num = 2 
batch_size= 20
keep_prob= 1
TARGET_STEP= 5
#wav=20

def minibatch(size):       
    randa=int(np.round(split*random.random()-1))
    lej=listtrain[randa].shape[1]
    randb=int(np.round((lej-(TIME_STEP+TARGET_STEP))*random.random()))
    bat=listtrain[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
    bat=bat.transpose()
    bat=bat[np.newaxis,:]
    batl=labeltrain[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
    batl=batl.transpose()
    batl=batl[np.newaxis,:]    
    for i in range(size-1): 
        randa=int(np.round(split*random.random()-1))
        lej=listtrain[randa].shape[1]
        randb=int(np.round((lej-(TIME_STEP+TARGET_STEP))*random.random()))
        bati=listtrain[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
        bati=bati.transpose()
        bati=bati[np.newaxis,:]
        bat=np.concatenate((bat,bati),axis=0)
        batli=labeltrain[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
        batli=batli.transpose()
        batli=batli[np.newaxis,:]  
        batl=np.concatenate((batl,batli),axis=0)

    return bat,batl

def minibatch_test(size):       
    randa=int(np.round((num-split)*random.random()-1))
    lej=listtest[randa].shape[1]
    randb=int(np.round((lej-(TIME_STEP+TARGET_STEP))*random.random()))
    bat=listtest[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
    bat=bat.transpose()
    bat=bat[np.newaxis,:]
    batl=labeltest[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
    batl=batl.transpose()
    batl=batl[np.newaxis,:]    
    for i in range(size-1): 
        randa=int(np.round((num-split)*random.random()-1))
        lej=listtest[randa].shape[1]
        randb=int(np.round((lej-(TIME_STEP+TARGET_STEP))*random.random()))
        bati=listtest[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
        bati=bati.transpose()
        bati=bati[np.newaxis,:]
        bat=np.concatenate((bat,bati),axis=0)
        batli=labeltest[randa][:,randb:randb+TIME_STEP+TARGET_STEP]
        batli=batli.transpose()
        batli=batli[np.newaxis,:]  
        batl=np.concatenate((batl,batli),axis=0)

    return bat,batl
# tensorflow placeholders
tf.reset_default_graph()
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])       
#tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, OUT_SIZE])        
tf_xa = tf.placeholder(tf.float32, [None, TARGET_STEP, INPUT_SIZE])       
tf_ya = tf.placeholder(tf.float32, [None, TARGET_STEP, OUT_SIZE])       
#
#w_wav=tf.Variable(tf.random_normal([TIME_STEP,wav]))
#scale_wav=tf.Variable(tf.random_normal([1,wav]))
#tao_wav=tf.Variable(tf.random_normal([1,wav]))
#w2_wav=tf.Variable(tf.random_normal([wav,TIME_STEP]))

w_in=tf.Variable(tf.random_normal([INPUT_SIZE,CELL_SIZE]))
b_in=tf.Variable(tf.random_normal([1,CELL_SIZE]))
w_out=tf.Variable(tf.random_normal([CELL_SIZE,OUT_SIZE]))
b_out=tf.Variable(tf.random_normal([OUT_SIZE]))
w_att=tf.Variable(tf.random_normal([TIME_STEP, TARGET_STEP]))
b_att=tf.Variable(tf.random_normal([TARGET_STEP]))
w_ind=tf.Variable(tf.random_normal([INPUT_SIZE+CELL_SIZE,CELL_SIZE]))
b_ind=tf.Variable(tf.random_normal([1,CELL_SIZE]))
w_bi=tf.Variable(tf.random_normal([1]))
w_bi2=tf.Variable(tf.random_normal([1]))
w_bid=tf.Variable(tf.random_normal([1]))
w_bid2=tf.Variable(tf.random_normal([1]))

# RNN'
#tf_xw=tf.transpose(tf_x,perm=[0, 2, 1])
#tf_xw2D=tf.reshape(tf_xw,[-1,TIME_STEP])  
#tf_xw2D=tf.matmul(tf_xw2D,w_wav)
#tf_xw=tf.reshape(tf_xw2D,[-1,INPUT_SIZE,wav])  
#
#tf_xw=(tf_xw-scale_wav)/tao_wav
## Morlet
#tf_xw=tf.exp(-tf.square(tf_xw))*tf.cos(1.75*tf_xw)
#tf_xw2D=tf.reshape(tf_xw,[-1,wav])  
#tf_xw2D=tf.matmul(tf_xw2D,w2_wav)
#tf_xw=tf.reshape(tf_xw2D,[-1,INPUT_SIZE,TIME_STEP])  
#
#tf_xw=tf.transpose(tf_xw,perm=[0, 2, 1])

tf_x2D=tf.reshape(tf_x,[-1,INPUT_SIZE])  
tf_x2D_rnn=tf.matmul(tf_x2D,w_in)+b_in
tf_x2D_rnn=tf.reshape(tf_x2D_rnn,[-1,TIME_STEP,CELL_SIZE]) 

stacked_rnn = []
for iiLyr in range(layer_num):   
    stacked_rnn.append(tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
#    stacked_rnn.append(tf.nn.rnn_cell.GRUCell(CELL_SIZE,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))

mcell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True) 
mcell = tf.nn.rnn_cell.DropoutWrapper(mcell, output_keep_prob=keep_prob)
#mcell=tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE)
init_s = mcell.zero_state(batch_size=batch_size, dtype=tf.float32)    # very first hidden state

stacked_rnn2 = []
for iiLyr2 in range(layer_num):   
#    stacked_rnn2.append(tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
    stacked_rnn2.append(tf.nn.rnn_cell.GRUCell(CELL_SIZE,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))

mcell2 = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn2, state_is_tuple=True) 
mcell2 = tf.nn.rnn_cell.DropoutWrapper(mcell2, output_keep_prob=keep_prob)
#mcell=tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE)
init_s2 = mcell2.zero_state(batch_size=batch_size, dtype=tf.float32)    # very first hidden state

outputs, final_s = tf.nn.bidirectional_dynamic_rnn(
    mcell,      
    mcell2,             # cell you have chosen
    tf_x2D_rnn,                       # input
    initial_state_fw=init_s,
    initial_state_bw=init_s2,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
outs2D = tf.reshape(outputs[0], [-1, CELL_SIZE]) 
outs2D2 = tf.reshape(outputs[1], [-1, CELL_SIZE]) 
outs2D_con=w_bi*outs2D+w_bi2*outs2D2
outputs=tf.reshape(outs2D_con,[batch_size, TIME_STEP, CELL_SIZE])
outputs=tf.transpose(outputs,perm=[0, 2, 1])

outs2D = tf.reshape(outputs, [-1, TIME_STEP]) 
#w_att=tf.nn.softmax(w_att)
attention2D=tf.matmul(outs2D, w_att)+b_att
attention=tf.reshape(attention2D, [batch_size, TARGET_STEP, CELL_SIZE])
pred_in=tf.concat([attention,tf_xa], axis=2)    
pred_in=tf.transpose(pred_in,perm=[0, 2, 1])
pre_size=INPUT_SIZE+CELL_SIZE
tf_xa2D=tf.reshape(pred_in,[-1,pre_size])  
tf_xa2D_rnn=tf.matmul(tf_xa2D,w_ind)+b_ind
tf_xa2D_rnn=tf.reshape(tf_xa2D_rnn,[-1,TARGET_STEP,CELL_SIZE]) 

stacked_rnnd = []
for iiLyr in range(layer_num):   
#    stacked_rnnd.append(tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
    stacked_rnnd.append(tf.nn.rnn_cell.GRUCell(CELL_SIZE,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))

mcelld = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnnd, state_is_tuple=True) 
mcelld = tf.nn.rnn_cell.DropoutWrapper(mcelld, output_keep_prob=keep_prob)
#mcell=tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE)
init_sd = mcelld.zero_state(batch_size=batch_size, dtype=tf.float32)    # very first hidden state
stacked_rnnd2 = []
for iiLyrd2 in range(layer_num):   
#    stacked_rnnd2.append(tf.nn.rnn_cell.BasicLSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
    stacked_rnnd2.append(tf.nn.rnn_cell.GRUCell(CELL_SIZE,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))

mcelld2 = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnnd2, state_is_tuple=True) 
mcelld2 = tf.nn.rnn_cell.DropoutWrapper(mcelld2, output_keep_prob=keep_prob)
#mcell=tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE)
init_sd2 = mcelld2.zero_state(batch_size=batch_size, dtype=tf.float32)    # very first hidden state

outputds, final_ds = tf.nn.bidirectional_dynamic_rnn(
    mcelld,      
    mcelld2,             # cell you have chosen
    tf_xa2D_rnn,                       # input
    initial_state_fw=init_sd,
    initial_state_bw=init_sd2,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
outs2Dd = tf.reshape(outputds[0], [-1, CELL_SIZE]) 
outs2Dd2 = tf.reshape(outputds[1], [-1, CELL_SIZE]) 
outs2dD_con=w_bi*outs2Dd+w_bi2*outs2Dd2
pred=tf.matmul(outs2dD_con ,w_out)+b_out

#net_outs2D = tf.layers.dense(outs2D, OUT_SIZE)
outs = tf.reshape(pred, [-1, TARGET_STEP, OUT_SIZE])          # reshape back to 3D

loss = tf.losses.mean_squared_error(labels=tf_ya, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

mse_e=[]
mae_e=[]
mse_t=[]
mae_t=[]
epoch=100
for i in range(epoch):
    
    losses=[]
    pred_num=[]
    sta_step=0
    ite=100
    for step in range(ite):
        
        dbtrain, lbtrain=minibatch(batch_size)
        feed_dict = {tf_x: dbtrain[:,0:TIME_STEP], tf_xa: dbtrain[:,0:TARGET_STEP], tf_ya: lbtrain[:,0:TARGET_STEP]}
        _, pred_, final_s_ , _loss= sess.run([train_op, outs, final_s, loss], feed_dict)     # train
        losses.append(_loss)
        sta_one=np.sum(abs(pred_-lbtrain[:,0:TARGET_STEP]))
        sta_step=sta_step+sta_one
        
#        plt.axis([0,TIME_STEP,0,1])
#        steps=np.linspace(0,TARGET_STEP+1,TARGET_STEP,dtype=np.float32)
#        plt.plot(steps, lbtrain[0,0:TARGET_STEP], 'r-',label='real data'); plt.plot(steps, pred_[0,0:TARGET_STEP], 'b-', label='prediction')
##        plt.plot(steps_origin, use_irr.flatten(), 'g-')
#        plt.legend()
#        plt.draw(); plt.pause(0.005)
    
    mse_emean=np.mean(losses)
    mae_emean=sta_step/(batch_size*TARGET_STEP*ite)
    print(i,'train_mse',mse_emean)
    print(i,'train_mae',mae_emean)    
    mse_e.append(mse_emean)
    mae_e.append(mae_emean)
    mae_te=[]
    mse_te=[]
    
    testite=200
    for step in range(testite):
        
        dbtest, lbtest=minibatch_test(batch_size)
        feed_dict = {tf_x: dbtest[:,0:TIME_STEP], tf_xa: dbtest[:,0:TARGET_STEP], tf_ya: lbtest[:,0:TARGET_STEP]}
        pred_, final_s_ , _loss= sess.run([outs, final_s, loss], feed_dict)     # train
        mse_te.append(_loss)
        sta_one=np.sum(abs(pred_-lbtest[:,0:TARGET_STEP]))
        mae_te.append(sta_one)
#        plt.axis([0,TIME_STEP,0,1])
#        steps=np.linspace(0,TARGET_STEP+1,TARGET_STEP,dtype=np.float32)
#        plt.plot(steps, lbtest[0,0:TARGET_STEP], 'r-',label='real data'); plt.plot(steps, pred_[0,0:TARGET_STEP], 'b-', label='prediction')
##        plt.plot(steps_origin, use_irr.flatten(), 'g-')
#        plt.legend()
#        plt.draw(); plt.pause(0.005)
    
    mse_tmean=np.mean(mse_te)
    mae_tmean=np.mean(mae_te)/(batch_size*TARGET_STEP
                     )
    print(i,'test_mse',mse_tmean)
    print(i,'test_mae',mae_tmean)       
    mse_t.append(mse_tmean)
    mae_t.append(mae_tmean)



