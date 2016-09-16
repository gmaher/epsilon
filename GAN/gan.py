import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath("../.."))
#import tensorflow_util.util as util_tf

sys.path.append(os.environ['DATASETS_PATH'])
from mnist.mnist import get_mnist

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Convolution2D, UpSampling2D, Flatten, Reshape, merge, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm

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
NBATCH = 32
GNfilters = 80
DNfilters = 30
W = 3
lrg = 2e-4
lrd = 3e-4
p = 0.5
leak = 0.2
adamg = Adam(lr=lrg)
adamd = Adam(lr=lrd)

#Generator
z = Input(shape=(NOISE_DIM,))
g = Dense(Npixels**2, activation='relu')(z)
g = BatchNormalization(mode=2)(g)
g = Reshape((Npixels,Npixels,1))(g)
g = Convolution2D(GNfilters,W,W, activation='relu', border_mode='same')(g)
g = BatchNormalization(mode=2)(g)
g = Convolution2D(GNfilters/4,W,W, activation='relu', border_mode='same')(g)
g = BatchNormalization(mode=2)(g)
g = Convolution2D(1,W,W, activation='sigmoid', border_mode='same')(g)

Generator = Model(z,g)
Generator.compile(optimizer=adamg,loss='binary_crossentropy')

#Discriminator
x = Input(shape=(Npixels,Npixels,1))
d = Convolution2D(DNfilters,W,W,activation='linear')(x)
d = LeakyReLU(leak)(d)
d = Dropout(p)(d)
d = Convolution2D(DNfilters/2,W,W,activation='linear')(d)
d = LeakyReLU(leak)(d)
d = Dropout(p)(d)
d = Flatten()(d)
d = Dense(2, activation='softmax')(d)

Discriminator = Model(x,d)
Discriminator.compile(optimizer=adamd, loss='categorical_crossentropy', metrics=['accuracy'])

#Adversarial classifier
gen_input = Input(shape=(NOISE_DIM,))
h = Generator(gen_input)
out_fake = Discriminator(h)
Discriminator.trainable = False
Gen_classifier = Model(gen_input,out_fake)
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
Discriminator.fit(X,y,nb_epoch=1,batch_size=100)

#calculate accuracy
def calculate_accuracy(model,X,y):
    yhat = model.predict(X)
    acc = np.mean(np.argmax(yhat,axis=1) == np.argmax(y,axis=1))
    return acc

#calculate_accuracy(X,y)

#Adversarial training
Niter = 1000000

Yd = np.zeros((2*NBATCH,2))
Yd[:NBATCH,0]=1
Yd[NBATCH:,1]=1

Yg = np.zeros((NBATCH,2))
Yg[:,0]=1

losses = {}
losses['g'] = []
losses['d'] = []
losses['d_gan'] = []
for i in tqdm(range(0,Niter)):
    z_batch = np.random.rand(NBATCH,NOISE_DIM)

    x_gan = Generator.predict(z_batch,batch_size=NBATCH)
    x_batch = get_batch(data_norm,NBATCH)

    X = np.vstack((x_batch,x_gan))

    #ld = Discriminator.train_on_batch(x_batch,Yd[:NBATCH,:])
    #ld2 = Discriminator.train_on_batch(x_gan, Yd[NBATCH:,:])
    ld = Discriminator.train_on_batch(X,Yd)
    losses['d'].append(ld[0])
    #losses['d_gan'].append(ld2[0])


    z_batch = np.random.rand(NBATCH,NOISE_DIM)
    lg = Gen_classifier.train_on_batch(z_batch,Yg)
    losses['g'].append(lg)

print "saving models"
Discriminator.save('Discriminator.h5')
Generator.save('Generator.h5')
Gen_classifier.save('Gen_classifier.h5')

plt.plot(losses['d'], color='red', linewidth=2, label='discriminator')
plt.plot(losses['g'], color='green', linewidth=2, label='generator')
#plt.plot(losses['d_gan'], color='blue', linewidth=2, label='discriminator_gan')

plt.legend()
plt.plot()
plt.show()

def plot_gen(generator, noise_dim, n_ex=16, dim=(4,4), figsize=(10,10)):
    noise = np.random.rand(n_ex,noise_dim)
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,0]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_gen(Generator, NOISE_DIM)
