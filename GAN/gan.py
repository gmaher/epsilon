import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath("../.."))
import tensorflow_util.util as util_tf

sys.path.append(os.environ['DATASETS_PATH'])
from mnist.mnist import get_mnist

from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, UpSampling2D, Flatten, Reshape, merge, BatchNormalization
from keras.optimizers import Adam

###########################################
# Get MNIST Data
###########################################
datasets = get_mnist()
data, _ = datasets[0]

NUM_DATA = data.shape[0]
Npixels=28
Nchannels=1

data = data.reshape((NUM_DATA, Npixels, Npixels,1))
data_norm = (data-np.amin(data))/(np.amax(data)-np.amin(data))

print "data.shape={}, data[0,:].shape={}".format(
    data_norm.shape,data_norm[0,:].shape
)
print "max(data)={}, min(data)={}".format(
    np.amax(data_norm), np.amin(data_norm)
)

###########################################
# Set up Keras models
###########################################
NOISE_DIM = 100
NBATCH = 25
Nfilters = 8
W = 3
lrg = 1e-4
lrd = 1e-3

adamg = Adam(lr=lrg)
adamd = Adam(lr=lrd)

#Generator
z = Input(shape=(NOISE_DIM,))
g = Dense(Npixels**2, activation='relu')(z)
g = Reshape((Npixels,Npixels,1))(g)
g = Convolution2D(Nfilters,W,W, activation='relu', border_mode='same')(g)
g = Convolution2D(1,W,W, activation='linear', border_mode='same')(g)

Generator = Model(z,g)
Generator.compile(optimizer=adamg,loss='binary_crossentropy')

#Discriminator
x = Input(shape=(Npixels,Npixels,1))
d = Convolution2D(Nfilters,W,W,activation='relu')(x)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,W,W,activation='relu')(d)
d = BatchNormalization(mode=2)(d)
d = Convolution2D(Nfilters,W,W,activation='relu')(d)
d = BatchNormalization(mode=2)(d)
d = Flatten()(d)
d = Dense(2, activation='softmax')(d)

Discriminator = Model(x,d)
Discriminator.compile(optimizer=adamd, loss='categorical_crossentropy', metrics=['accuracy'])

#Adversarial classifier
h = Generator(z)
out_fake = Discriminator(h)
Discriminator.trainable = False
Gen_classifier = Model(z,out_fake)
Gen_classifier.compile(optimizer=adamg, loss='categorical_crossentropy')

###########################################
# Training
##########################################
def get_batch(xdata, nbatch):
    N = xdata.shape[0]
    inds = np.random.choice(N, size=nbatch, replace=False)
    xret = xdata[inds,:]

    return xret

#Pretrain discriminator
num_batch = 100
z_pre = np.random.rand(num_batch*NBATCH,NOISE_DIM)
x_pre = get_batch(data_norm,num_batch*NBATCH)
x_gan = Generator.predict(z_pre,batch_size=num_batch*NBATCH)
X = np.vstack((x_pre,x_gan))
y = np.zeros((X.shape[0],2))

y[:X.shape[0]/2,0]=1
y[X.shape[0]/2:,1]=1

print "x_pre.shape={}, x_gan.shape={}, X.shape={}, y.shape={}".format(
    x_pre.shape, x_gan.shape, X.shape, y.shape
)

Discriminator.trainable = True
Discriminator.fit(X,y,nb_epoch=1,batch_size=32)

#calculate accuracy
def calculate_accuracy(X,y):
    yhat = Discriminator.predict(X)
    acc = np.mean(np.amax(yhat,axis=1) == np.amax(y,axis=1))
    print "Discriminator accuracy={}".format(acc)
    return acc

calculate_accuracy(X,y)

#Adversarial training
Niter = 100

Yd = np.zeros((2*NBATCH,2))
Yd[:NBATCH,0]=1
Yd[NBATCH:,1]=1

Yg = np.zeros((NBATCH,2))
Yg[:,0]=1

print_step = 10

for i in range(0,Niter):
    z_batch = np.random.rand(NBATCH,NOISE_DIM)

    x_gan = Generator.predict(z_batch,batch_size=NBATCH)
    x_batch = get_batch(data_norm,NBATCH)

    X = np.vstack((x_batch,x_gan))

    ld = Discriminator.train_on_batch(X,Yd)

    z_batch = np.random.rand(NBATCH,NOISE_DIM)
    lg = Gen_classifier.train_on_batch(z_batch,Yg)

    if (i % print_step) ==0:
        print "Discriminator loss={}, Generator loss={}".format(ld, lg)

calculate_accuracy(X,Yd)
