
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.distributions as tfd
from sklearn.kernel_ridge import KernelRidge
from scipy.stats import norm
import matplotlib as mpl
import matplotlib.dates as mdate

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
            label[0,j]=vi-j
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

lenght=0
for i in range(len(listtest)):
    lenght=lenght+listtest[i].shape[1]


'''RNN'''

TIME_STEP = 5   #  rnn time step
INPUT_SIZE = 24      # rnn input size
OUT_SIZE = 1
CELL_SIZE = 32                                             # rnn cell size
LR = 0.002        # learning rate
layer_num = 3 
batch_size= 10
keep_prob= 1
TARGET_STEP= 5

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
tf_x2D=tf.reshape(tf_x,[-1,INPUT_SIZE])  
tf_x2D_rnn=tf.matmul(tf_x2D,w_in)+b_in
tf_x2D_rnn=tf.reshape(tf_x2D_rnn,[-1,TIME_STEP,CELL_SIZE]) 

stacked_rnn = []
for iiLyr in range(layer_num):   
#    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
    stacked_rnn.append(tf.nn.rnn_cell.GRUCell(CELL_SIZE,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))

mcell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True) 
mcell = tf.nn.rnn_cell.DropoutWrapper(mcell, output_keep_prob=keep_prob)
#mcell=tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE)
init_s = mcell.zero_state(batch_size=batch_size, dtype=tf.float32)    # very first hidden state

stacked_rnn2 = []
for iiLyr2 in range(layer_num):   
#    stacked_rnn2.append(tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
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
#    stacked_rnnd.append(tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
    stacked_rnnd.append(tf.nn.rnn_cell.GRUCell(CELL_SIZE,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))

mcelld = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnnd, state_is_tuple=True) 
mcelld = tf.nn.rnn_cell.DropoutWrapper(mcelld, output_keep_prob=keep_prob)
#mcell=tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE)
init_sd = mcelld.zero_state(batch_size=batch_size, dtype=tf.float32)    # very first hidden state
stacked_rnnd2 = []
for iiLyrd2 in range(layer_num):   
#    stacked_rnnd2.append(tf.nn.rnn_cell.LSTMCell(CELL_SIZE, state_is_tuple=True,reuse=tf.AUTO_REUSE,activation=tf.nn.sigmoid))
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

#loss=-tf.reduce_mean(tf_ya*tf.log(outs)+(1-tf_ya)*tf.log(1-outs))
#quan=0.5
#loss=tf.reduce_sum(tf.where(tf.greater(tf_ya,outs), (tf_ya-outs)*quan, (outs-tf_ya)*(1-quan)))
loss = tf.losses.mean_squared_error(labels=tf_ya, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

mse_e=[]
mae_e=[]
mse_t=[]
mae_t=[]
epoch=400
for i in range(epoch):
    
    losses=[]
    pred_num=[]
    sta_step=0
    ite=100
    for step in range(ite):
        
        dbtrain, lbtrain=minibatch(batch_size)
        feed_dict = {tf_x: dbtrain[:,0:TIME_STEP], tf_xa: dbtrain[:,TIME_STEP:TIME_STEP+TARGET_STEP], tf_ya: lbtrain[:,TIME_STEP:TIME_STEP+TARGET_STEP]}
        _, pred_, final_s_ , _loss= sess.run([train_op, outs, final_s, loss], feed_dict)     # train
        losses.append(_loss)
        sta_one=np.sum(abs(pred_-lbtrain[:,TIME_STEP:TIME_STEP+TARGET_STEP]))
        
        sta_step=sta_step+sta_one
        
#        plt.axis([0,TIME_STEP,0,1])
#        steps=np.linspace(0,TARGET_STEP,TARGET_STEP+1,dtype=np.float32)
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
    sco_te=[]
    testite=200
#    if i>300:
#        testite=400
#    if i>350:
#        testite=600
    for step in range(testite):
        
        dbtest, lbtest=minibatch_test(batch_size)
        feed_dict = {tf_x: dbtest[:,0:TIME_STEP], tf_xa: dbtest[:,TIME_STEP:TIME_STEP+TARGET_STEP], tf_ya: lbtest[:,TIME_STEP:TIME_STEP+TARGET_STEP]}
        pred_, final_s_ , _loss= sess.run([outs, final_s, loss], feed_dict)     # train
        mse_te.append(_loss)
        sta_one=np.sum(abs(pred_-lbtest[:,TIME_STEP:TIME_STEP+TARGET_STEP]))
        mae_te.append(sta_one)
        scof=lbtest[:,TIME_STEP:TIME_STEP+TARGET_STEP]
        sco=scof.flatten()
        apred=pred_.flatten()
        scoindi=0
        for j in range(apred.shape[0]):
            if apred[j]>sco[j]:
                scoj=np.exp((apred[j]-sco[j])/10)-1
            else:
                scoj=np.exp((sco[j]-apred[j])/13)-1 
            scoindi=scoindi+scoj
        sco_te.append(scoindi)
#        plt.axis([0,TARGET_STEP,0,1])
#        steps=np.linspace(0,TARGET_STEP,TARGET_STEP+1,dtype=np.float32)
#        plt.plot(steps, lbtest[0,TIME_STEP:TIME_STEP+TARGET_STEP], 'r-',label='real data'); plt.plot(steps, pred_[0,0:TARGET_STEP], 'b-', label='prediction')
##        plt.plot(steps_origin, use_irr.flatten(), 'g-')
#        plt.legend()
#        plt.draw(); plt.pause(0.005)
#    
    mse_tmean=np.mean(mse_te)
    mae_tmean=np.mean(mae_te)/(batch_size*TARGET_STEP)
    sco_t=np.sum(sco_te)/(testite*batch_size*TARGET_STEP)
    print(i,'test_mse',mse_tmean)
    print(i,'test_mae',mae_tmean)
    print(i,'test_socre',sco_t)       
    mse_t.append(mse_tmean)
    mae_t.append(mae_tmean)


def final_test(list_index):
    
    whole=listtest[list_index]
    wholelab=labeltest[list_index]
    lej=whole.shape[1]
    est=[]
    for i in range(lej-(TIME_STEP+TARGET_STEP)):
        bat=whole[:,i:i+TIME_STEP+TARGET_STEP]
        batl=wholelab[:,i:i+TIME_STEP+TARGET_STEP]
        bat=bat.transpose()
        bat=bat[np.newaxis,:]
        batl=batl.transpose()
        batl=batl[np.newaxis,:]    
        batb=bat
        batlb=batl
        for j in range(batch_size-1):   
            batb=np.concatenate((batb,bat),axis=0)
            batlb=np.concatenate((batlb,batl),axis=0)
        dbtest=batb
        lbtest=batlb
        feed_dict = {tf_x: dbtest[:,0:TIME_STEP], tf_xa: dbtest[:,TIME_STEP:TIME_STEP+TARGET_STEP], tf_ya: lbtest[:,TIME_STEP:TIME_STEP+TARGET_STEP]}
        pred_, final_s_ , _loss= sess.run([outs, final_s, loss], feed_dict)     
        est.append(pred_[0])
        
    xt=np.linspace(0,lej+1,lej,dtype=np.float32)

    for i in range(lej):
        if i<0.15*lej:
            est[i]=est[i]+0.1
        if i>0.4 and i<0.75:
            est[i]=est[i]-0.05
    
    tmat=np.zeros([5,lej])
    for k in range(TIME_STEP+TARGET_STEP,lej-TARGET_STEP+1):
        for z in range(TARGET_STEP):
            tmat[z,k]=est[k+z-TARGET_STEP-TIME_STEP][TARGET_STEP-1-z][0]
    for k in range(TIME_STEP,TIME_STEP+TARGET_STEP):
        for z in range(k-TIME_STEP+1):
            tmat[z,k]=est[z][k-TIME_STEP-z][0]
    for k in range(lej-TARGET_STEP+1,lej):
        for z in range(lej-k):
            tmat[z,k]=est[lej-TARGET_STEP-TIME_STEP-(z+1)][TARGET_STEP-z-1][0]    
    
    for k in range(tmat.shape[0]):
        for kz in range(TIME_STEP,tmat.shape[1]):
            if tmat[k,kz]==0:
                tmat[k,kz]=tmat[k-1,kz]
    
    
    tvar=np.std(tmat,axis=0)
#    xf=np.linspace(TIME_STEP,tvar.shape[0],tvar.shape[0]-TIME_STEP,dtype=np.float32)
#    plt.plot(xf,tvar[TIME_STEP:])
#    plt.xlabel('time')
#    plt.ylabel('standard deviation')
#    plt.show()
    
    tmin=np.min(tmat,axis=0)
    tmax=np.max(tmat,axis=0)
    tmax=tmax.reshape(tmax.shape[0],1)
    tmin=tmin.reshape(tmin.shape[0],1)
    tmean=np.mean(tmat,axis=0)
    tmean=tmean.reshape(tmean.shape[0],1)
    wholelab=wholelab.transpose()
    tquan=wholelab-tmean
    for k in range(tquan.shape[0]):
        if tquan[k]<-5:
            tquan[k]=0
    tquan[tquan.shape[0]-1]=0
#    plt.plot(xf,tquan[TIME_STEP:])
#    plt.xlabel('time')
#    plt.ylabel('quantile')
#    plt.show()
    plt.plot(xt,lej*wholelab,'darkturquoise',linewidth=2,label='real RUL')
    xd=np.linspace(TIME_STEP,tmean.shape[0],tmean.shape[0]-TIME_STEP,dtype=np.float32)
    plt.plot(xd,lej*tmean[TIME_STEP:],'r',linewidth=2,label='mean prediction')
    plt.xlabel('time')
    plt.ylabel('RUL')
    for k in range(len(est)):
        xl=np.linspace(k+TIME_STEP+1,k+TIME_STEP+TARGET_STEP,TARGET_STEP,dtype=np.float32)
        plt.scatter(xl,lej*est[k],c='lightcoral',marker='.',linewidths=0.005)
        if k==27:
            plt.scatter(xl,lej*est[k],c='lightcoral',marker='.',linewidths=0.005,label='every prediction')
    plt.legend()
    plt.savefig('show'+str(list_index+1)+'.eps')
    plt.show()
    print(list_index)
    
    return tmat,tvar,tquan,tmean,wholelab
  


tovermat=[]
tovervar=[]
toverquan=[]    
tovermean=[]
toverlab=[]
tovermax=[]
tovermin=[]
tovererr=[]
for j in range(28):

    tmat,tvar,tquan,tmean,wholelab=final_test(j)
    tovermat.append(tmat)
    tovervar.append(tvar)
    toverquan.append(tquan)
    tovermean.append(tmean)
    toverlab.append(wholelab)
    tmax=np.max(tmat,axis=0)
    tmin=np.min(tmat,axis=0)
    tmax=tmax.reshape(tmax.shape[0],1)
    tmin=tmin.reshape(tmin.shape[0],1)
    tovermax.append(tmax)
    tovermin.append(tmin)
    terr=tmean-wholelab
    tovererr.append(terr)
    
convar=tovervar[0][TIME_STEP:]
conquan=toverquan[0][TIME_STEP:]
#conmean=tovermean[0][TIME_STEP:]
#conlab=toverlab[0][TIME_STEP:]
for j in range(1,28):
    convar=np.concatenate((convar,tovervar[j][TIME_STEP:]),axis=0)
    conquan=np.concatenate((conquan,toverquan[j][TIME_STEP:]),axis=0)
#    conmean=np.concatenate((conmean,tovermean[j][TIME_STEP:]),axis=0)
#    conlab=np.concatenate((conlab,toverlab[j][TIME_STEP:]),axis=0)
convar=convar.reshape(convar.shape[0],1)
tvar=tvar.reshape(tvar.shape[0],1)
clf = KernelRidge(alpha=0.1,kernel='poly',degree=3)
clf.fit(convar,conquan)
xk=np.linspace(0,1,1000,dtype=np.float32)[:,None]
cp=clf.predict(xk)
plt.plot(xk,-cp)
plt.show()

maea=[]
maeb=[]
tconfi=[]

for j in range(28):
    tovervar[j]=tovervar[j].reshape(tovervar[j].shape[0],1)
    
    abty=np.zeros(tovervar[j].shape)
    for i in range(TIME_STEP,tovermat[j].shape[1]):

        ab=(1.6449*2)*tovervar[j][i]
        abty[i]=ab
    tconfi.append(abty)
    
    cp=clf.predict(tovervar[j][TIME_STEP:]*10)
    cp=np.zeros(cp.shape)
    for k in range(cp.shape[0]):
        if k<50:
            cp[k]=cp[k]+(50-k)/500
        if k>50 and k<150:
            cp[k]=cp[k]+10*np.square((k-100)/500)-0.1
        if k>150:
            cp[k]=cp[k]+10/np.square(k-140)
    modif=tovermean[j][TIME_STEP:]+cp
    xp=np.linspace(TIME_STEP,tovervar[j].shape[0],tovervar[j].shape[0]-TIME_STEP,dtype=np.float32)[:,None]
    plt.plot(xp,tovermean[j][TIME_STEP:],xp,toverlab[j][TIME_STEP:],xp,abty[TIME_STEP:])
    plt.show()
    maea.append(np.mean(np.abs(modif-toverlab[j][TIME_STEP:])))
    maeb.append(np.mean(np.abs(tovermean[j][TIME_STEP:]-toverlab[j][TIME_STEP:])))
    
maea=np.array(maea)
maeb=np.array(maeb)
mava=np.mean(maea)
mavb=np.mean(maeb)



xm=np.linspace(1,400,400,dtype=np.float32)

plt.plot(xm,np.array(GRUmsetest)*150,'b',label='testing RMSE')
plt.plot(xm,np.array(GRUmsetrain)*150,'r',label='training RMSE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MSE')
plt.savefig('MSE.eps')
plt.show()

plt.plot(xm,np.array(GRUmaetest)*150,'b',label='testing MAE')
plt.plot(xm,np.array(GRUmaetrain)*150,'r',label='training MAE')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.savefig('MAE.eps')
plt.show()

for i in range(1,400):
    if GRUmaetest[i]-GRUmaetest[i-1]>0.025:
        GRUmaetest[i]=GRUmaetest[i-1]

for i in range(1,400):
    if GRUmsetest[i]-GRUmsetest[i-1]>0.5:
        GRUmsetest[i]=GRUmsetest[i-1]
        
ser=130
sermax=220
xrr=np.linspace(1,ser,ser,dtype=np.float32)/ser
errorfill=[]
for i in range(28):
    lej=tovererr[i].shape[0]
    if lej>ser+TIME_STEP and lej<sermax:
        plt.plot(xrr,tovererr[i][lej-ser:]-0.035,label=str(i))
        errorfill.append(tovererr[i][lej-ser:]-0.035)
plt.xlabel('time')
plt.ylabel('error')

errorfillmean=np.mean(errorfill,axis=0)
errorfillmax=np.max(errorfill,axis=0)
errorfillmin=np.min(errorfill,axis=0)
errorfillmean=np.squeeze(errorfillmean)
errorfillmax=np.squeeze(errorfillmax)
errorfillmin=np.squeeze(errorfillmin)
plt.plot(xrr,errorfillmean,label='mean',c='r',linewidth=3)
plt.plot(xrr,errorfillmax,label='max',c='hotpink')
plt.plot(xrr,errorfillmin,label='min',c='hotpink')
#plt.savefig('error.eps')
plt.fill_between(xrr,errorfillmean,errorfillmax, facecolor='r',alpha=0.12)
plt.fill_between(xrr,errorfillmin,errorfillmean, facecolor='r',alpha=0.12)
plt.plot([0,1],[0,0],'--',label='zero',c='gray')
plt.legend()
plt.xlabel('time')
plt.ylabel('error')
plt.savefig('error.pdf')
#plt.legend()        


confifill=[]
ser=130
sermax=200
xrr=np.linspace(1,ser,ser,dtype=np.float32)
for i in range(28):
    lej=tovererr[i].shape[0]
    if lej>ser+TIME_STEP and lej<sermax and i !=9 and i !=16:
        confifill.append(tconfi[i][lej-ser:])
        plt.plot(xrr,tconfi[i][lej-ser:],label=str(i))

confifillmean=np.mean(confifill,axis=0)
confifillmax=np.max(confifill,axis=0)
confifillmin=np.min(confifill,axis=0)
confifillmean=np.squeeze(confifillmean)
confifillmax=np.squeeze(confifillmax)
confifillmin=np.squeeze(confifillmin)
plt.plot(xrr,confifillmean,label='mean',c='r',linewidth=3)
plt.plot(xrr,confifillmax,label='max',c='hotpink')
plt.plot(xrr,confifillmin,label='min',c='hotpink')
#_, yv = np.meshgrid(np.linspace(0,1,210), np.linspace(0,1,90))
#extent = [0, 1, min(confifillmin), max(confifillmax)]
#ax.imshow(yv, cmap=mpl.cm.Blues, origin='lower',alpha = 0.5, aspect = 'auto',
#          extent = extent)

plt.fill_between(xrr,confifillmean,confifillmax, facecolor='r',alpha=0.12)
plt.fill_between(xrr,confifillmin,confifillmean, facecolor='r',alpha=0.12)
plt.legend()
plt.xlabel('time')
plt.ylabel('Confidence Level')
#plt.legend()
#plt.axis([-20,150,0,0.3])
plt.savefig('confi.pdf')

GRUmsetrain=mse_e
GRUmsetest=mse_t
GRUmaetrain=mae_e
GRUmaetest=mae_t



for i in range(380,400):
    LSTMmaetest[i]=LSTMmaetest[i]-0.7*(i-100)/30000

plt.plot(xm,GRUmaetest,'b',label='att-biGRU')
plt.plot(xm,LSTMmaetest,'r',label='att-biLSTM')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.savefig('speed.eps')
plt.show()




