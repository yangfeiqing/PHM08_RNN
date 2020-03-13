

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.distributions as tfd
import random
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D

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
    listdata[i]=listdata[i][5:]
''' 218 data is splited as 180 training data and 38 testing data'''
split=180

listtrain=listdata[:180]
listtest=listdata[180:]
labeltrain=labeldata[:180]
labeltest=labeldata[180:]
#train_y=train_y[0:nose]

def rand_sel():       
    randa=int(np.round((num-split)*random.random()-1))
    
    return listtrain[randa],randa

hidden=400
input_size=21
code_size=3
lr=0.0001


def make_encoder(data, code_size):
    
    x = tf.layers.flatten(data)
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    
    return tfd.MultivariateNormalDiag(loc, scale), loc, scale

def make_prior(code_size):
#    produce guassian N(0,1)
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    
    return tfd.MultivariateNormalDiag(loc, scale)

def make_decoder(code, data_shape):
    
    x = code
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    logit = tf.layers.dense(x, np.prod(data_shape))
    logit = tf.reshape(logit, [-1] + data_shape)
    
    return tfd.Independent(tfd.Bernoulli(logit), 2)


make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)


data = tf.placeholder(tf.float32, [None, input_size, 1])

prior = make_prior(code_size)
posterior, loc, scale = make_encoder(data, code_size)
code_all = posterior.sample(10)
code = tf.reduce_mean(code_all,reduction_indices=0)

likelihood = make_decoder(code, [input_size, 1]).log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)

optimize = tf.train.AdamOptimizer(lr).minimize(-elbo)
samples = make_decoder(prior.sample(10), [input_size, 1]).mean()

init1 = tf.global_variables_initializer()  
sess1 = tf.Session()
sess1.run(init1)

saver = tf.train.Saver()  

if __name__ == '__main__':
    
    for epoch in range(5):
        
        mean_rec=[]
        samp_rec=[]
        for i in range(200):
            
            listindex,_=rand_sel()
            lej=listindex.shape[1]
#            print('i',i,'length',lej)
            for j in range(lej):
                train_y=listindex.transpose()
#                print(train_y.shape)
                train_y=train_y[:,:,np.newaxis]
                sess1.run(optimize, {data: train_y})

        for _ in range(5):

            listindex,rrr=rand_sel()
            lej=listindex.shape[1]
            for j in range(lej):
                train_y=listindex.transpose()
#                print(train_y.shape)
                train_y=train_y[:,:,np.newaxis]
                test_elbo, test_codes, test_samples, mean, var, alll = sess1.run(
                    [elbo, code, samples, loc, scale, code_all], {data: train_y})
                mean_rec.append(mean)
                samp_rec.append(test_codes)
                if lej-j<5:
                    print('Epoch', epoch, 'elbo', test_elbo)
                    plt.scatter(mean[:, 0], mean[:, 1])  
                    plt.axis([-1.5,1.5,-1.5,1.2])
                    plt.show()

saver.save(sess1, './vaephm_model') 

def train():                  

    for epoch in range(5):
        
        mean_rec=[]
        samp_rec=[]
        for i in range(200):
            
            listindex,_=rand_sel()
            lej=listindex.shape[1]
#            print('i',i,'length',lej)
            for j in range(lej):
                train_y=listindex.transpose()
#                print(train_y.shape)
                train_y=train_y[:,:,np.newaxis]
                sess1.run(optimize, {data: train_y})

        for _ in range(5):

#            listindex,rrr=rand_sel()
            listindex=listdata[13]
            lej=listindex.shape[1]
            for j in range(lej):
                train_y=listindex.transpose()
#                print(train_y.shape)
                train_y=train_y[:,:,np.newaxis]
                test_elbo, test_codes, test_samples, mean, var, alll = sess1.run(
                    [elbo, code, samples, loc, scale, code_all], {data: train_y})
                mean_rec.append(mean)
                samp_rec.append(test_codes)
                if lej-j<5:
                    print('Epoch', epoch, 'elbo', test_elbo)
#                    plt.scatter(mean[:, 0], mean[:, 1])  
#                    plt.xlabel('feature 1')
#                    plt.ylabel('feature 2')
#                    plt.savefig('vae2.eps')
#                    plt.show()
                    
                    ax=plt.subplot(111,projection='3d') 
                    ax.scatter(mean[:,0], mean[:,1], mean[:,2],c='r')
                    ax.set_xlabel('feature 1')
                    ax.set_ylabel('feature 2')
                    ax.set_zlabel('feature 3')
                    plt.savefig('vae3.eps')
                    plt.show()
                    
def get_vae_whole(inda):
    
    meanout=[]
    sampout=[]
    timess=listtrain[inda]
    timess=timess.transpose()
    timess=timess[:,:,np.newaxis]
    test_codes, mean, var= sess1.run(
                    [code, loc, scale], {data: timess})        
    meanout.append(mean)
    sampout.append(test_codes)
    
    return meanout, sampout

def get_vae_series(X):
    
    batch=X.shape[0]
#    print(batch)
    for i in range(batch):
        
        X_bat=X[i]
        X_bat=X_bat[:,:,np.newaxis]
        meanout=np.zeros([X.shape[0],X.shape[1],code_size])
        sampout=np.zeros([X.shape[0],X.shape[1],code_size])
        test_codes, mean, var= sess1.run(
                    [code, loc, scale], {data: X_bat})        
        for k in range(mean.shape[0]):
            mean[k]=(mean[k]-np.mean(mean[k]))/np.std(mean[k])
            test_codes[k]=(test_codes[k]-np.mean(test_codes[k]))/np.std(test_codes[k])
        meanout[i]=mean
        sampout[i]=test_codes    
    
    return meanout, sampout



xp=np.linspace(0,mean.shape[0],mean.shape[0],dtype=np.float32)
plt.plot(xp,mean[:,1])



t7=[]
for i in range(mean.shape[0]):
    if mean[i,0]>0.5:
        t7.append(train_y[i,:])
t7=np.array(t7)
t7=t7.reshape(t7.shape[0],21)

xd=np.linspace(0,t7.shape[0],t7.shape[0],dtype=np.float32)
for j in range(t7.shape[1]):
    if np.max(t7[:,j])-np.min(t7[:,j])>0:       
        t7[:,j]=(t7[:,j]-np.min(t7[:,j]))/(np.max(t7[:,j])-np.min(t7[:,j]))
    




    plt.plot(xd,t7[:,1],label=str(2))
    plt.plot(xd,t7[:,3],label=str(4))
    plt.plot(xd,t7[:,7],label=str(8))  
    plt.plot(xd,t7[:,8],label=str(9))  
    plt.plot(xd,t7[:,10],label=str(11))  
    plt.plot(xd,t7[:,12],label=str(13))
    plt.plot(xd,t7[:,14],label=str(15)) 

#    plt.scatter(xd,t7[:,1],'o')
    plt.legend()
    plt.xlabel('time step in biggest class')
    plt.ylabel('measurements after normalization')
    plt.savefig('sensor7.eps')
    plt.show()
    
        
        
ao=[]
co=[]
mo=[]
meanall=[]

for i in range(180):
    
    mean,_=get_vae_whole(i)
    mean=np.array(mean)
    mean=mean.reshape(mean.shape[1],mean.shape[2])
    meanall.append(mean)
    t7=[]
    for j in range(mean.shape[0]):
        if mean[j,0]>0.5:
            t7.append(listdata[i][:,j])
    t7=np.array(t7)
#    print(t7.shape)
    t=np.linspace(1,t7.shape[0],t7.shape[0],dtype=np.float32)
    cs=[]
    ms=[]
    for k in range(21):
        ca=np.abs(np.sum((t7[:,k]-np.mean(t7[:,k]))*(t-np.mean(t))))
        cb=np.sqrt(np.sum(np.square(t7[:,k]-np.mean(t7[:,k]))*np.square(t-np.mean(t))))
        if cb==0:
            c=0
        else:
            c=ca/cb
        cs.append(c)
        mf=0
        mi=0
        for z in range(t7.shape[0]-1):
            if t7[z+1,k]-t7[z,k]>0:
                mf=mf+1
            if t7[z+1,k]-t7[z,k]<0:
                mi=mi+1        
        m=np.abs(mf-mi)/(t7.shape[0]-1)
        ms.append(m)
    cs=np.array(cs)
    ms=np.array(ms)
    ass=(cs+ms)/2
    
    ao.append(ass)
    co.append(cs)
    mo.append(ms)
    
ao=np.array(ao)
co=np.array(co)
mo=np.array(mo)
meanall=np.array(meanall)

afinal=np.mean(ao,axis=0)
cfinal=np.mean(co,axis=0)
mfinal=np.mean(mo,axis=0)

np.save('vae_mean.npy',meanall)
np.save('vae_ao.npy',ao)
np.save('vae_co.npy',co)
np.save('vae_mo.npy',mo)

at=np.linspace(1,21,21,dtype=np.float32)
plt.bar(range(1,len(afinal)+1), afinal)
plt.plot(at,afinal,'r',label='index')
plt.scatter(at,afinal,marker='s')
plt.legend()
plt.axis([0,22,0,2.3])
plt.xlabel('sensor number')
plt.ylabel('index')
plt.savefig('selection.eps')
plt.show()