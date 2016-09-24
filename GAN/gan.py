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
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm

def add_noise_to_weights(model, eps=0.1):
    weight_list = model.get_weights()
    new_weights = []

    for w in weight_list:
        noise = np.random.normal(size=w.shape)*eps

        new_weights.append(w+noise)

    return new_weights

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
NBATCH = 20
GNfilters = 60
DNfilters = 20
W = 3
lrg = 1e-3
lrd = 5e-3
p = 0.75
leak = 0.2
optg = Adam(lr=lrg)
optd = Adam(lr=lrd)
#optg = SGD(lr=lrg)
#optd = SGD(lr=lrd)
EPS = 0.01
decay = 0.55
Niter = 2000
use_noise = 1
use_replay = 0
#########################
# Experience Replay
#########################
D = []
T = 20 #once every T iterations perform experience Replay
N = 10 #perform N experience replay updates

#Generator
z = Input(shape=(NOISE_DIM,))
g = Dense((Npixels/4)**2, activation='relu')(z)
g = BatchNormalization(mode=2)(g)
g = Reshape((Npixels/4,Npixels/4,1))(g)
g = Convolution2D(GNfilters,W,W, activation='relu', border_mode='same')(g)
g = BatchNormalization(mode=2)(g)
g = UpSampling2D(size=(4,4))(g)
g = Convolution2D(GNfilters/2,W,W, activation='relu', border_mode='same')(g)
g = BatchNormalization(mode=2)(g)
g = Convolution2D(1,W,W, activation='sigmoid', border_mode='same')(g)

Generator = Model(z,g)
Generator.compile(optimizer=optg,loss='binary_crossentropy')

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
Discriminator.compile(optimizer=optd, loss='categorical_crossentropy', metrics=['accuracy'])

#Adversarial classifier
gen_input = Input(shape=(NOISE_DIM,))
h = Generator(gen_input)
out_fake = Discriminator(h)
Discriminator.trainable = False
Gen_classifier = Model(gen_input,out_fake)
Gen_classifier.compile(optimizer=optg, loss='categorical_crossentropy')

Discriminator.trainable = True
print "Discriminator params={}\n Generator params={}\n Ratio={}".format(
    Discriminator.count_params(), Generator.count_params(),
    float(Discriminator.count_params())/Generator.count_params()
)

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

Discriminator.fit(X,y,nb_epoch=1,batch_size=100)

#calculate accuracy
def calculate_accuracy(model,X,y):
    yhat = model.predict(X)
    acc = np.mean(np.argmax(yhat,axis=1) == np.argmax(y,axis=1))
    return acc

#calculate_accuracy(X,y)

#Adversarial training

Yd = np.zeros((2*NBATCH,2))
Yd[:NBATCH,0]=1
Yd[NBATCH:,1]=1

Yg = np.zeros((NBATCH,2))
Yg[:,0]=1

losses = {}
losses['g'] = []
losses['d'] = []
losses['d_replay'] = []
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

    #Add noise to both models
    if use_noise == 1:
        perturb = 1.0/((1.0+i)**decay)*EPS
        Discriminator.set_weights(add_noise_to_weights(Discriminator,eps=perturb))
        Generator.set_weights(add_noise_to_weights(Generator,eps=perturb))

    if use_replay == 1:
        #Experience Replay
        D.append((X,Yd))
        if i % T == 0:
            for k in range(0,N):
                ind = np.random.randint(len(D))
                X_replay, Y_replay = D[ind]
                ld_replay = Discriminator.train_on_batch(X_replay, Y_replay)
                losses['d_replay'].append(ld_replay[0])

#print "Final noise = {}".format(perturb)
print "saving models"
Discriminator.save('Discriminator.h5')
Generator.save('Generator.h5')
Gen_classifier.save('Gen_classifier.h5')

plt.plot(losses['d'], color='red', linewidth=2, label='discriminator')
plt.plot(losses['d_replay'], color='blue', linewidth=2, label='discriminator replay')
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
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

plot_gen(Generator, NOISE_DIM)
