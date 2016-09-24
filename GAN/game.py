import numpy as np
import sys
import os

sys.path.append("/home/gabriel/projects/tcl_code")

from utility import *

def adversarial_update(x,y,lr,n):
    x = x - lr*y
    y = y - lr*(-x)
    return x,y

def make_noise_update(noise=0.1):
    def noise_update(x,y,lr,n):
        x,y = adversarial_update(x,y,lr,n)
        x = x + noise*np.random.randn()
        y = y + noise*np.random.randn()
        return x,y
    return noise_update

def make_ascent_update(prob=0.1):
    def ascent_update(x,y,lr,n):
        xn,yn = adversarial_update(x,y,lr,n)
        dx = xn-x
        dy = yn-y
        px = np.random.rand()
        py = np.random.rand()
        if px > prob:
            x = xn
        else:
            x = x-dx

        if py > prob:
            y = yn
        else:
            y = y-dy

        return x,y
    return ascent_update

def make_decay_noise_update(noise=0.1,decay=100):
    def decay_noise_update(x,y,lr,n):
        x,y = adversarial_update(x,y,lr,n)

        d = float(decay)/(decay+n)
        x = x + decay*noise*np.random.randn()
        y = y + decay*noise*np.random.randn()
        return x,y

    return decay_noise_update


def sim_game(n, update, lr=0.1):
    x = 4*(2*np.random.rand()-1)
    y = 4*(2*np.random.rand()-1)

    xs = []
    ys = []
    xs.append(x)
    ys.append(y)
    for i in range(0,n):

        x,y = update(x,y,lr,n)

        xs.append(x)
        ys.append(y)

    return xs,ys

noise_update = make_noise_update(0.2)
ascent_update = make_ascent_update(0.3)
decay_noise_update = make_decay_noise_update(noise=0.001,decay=20)

Ngames = 4
steps = 1000
Xs = []
Ys = []
leg = []
for i in range(0,Ngames):
    #xl,yl = sim_game(steps, adversarial_update, lr=0.1)
    #xl,yl = sim_game(steps, noise_update, lr=0.1)
    xl,yl = sim_game(steps, ascent_update, lr=0.1)
    #xl,yl = sim_game(steps, decay_noise_update, lr=0.1)
    Xs.append(xl)
    Ys.append(yl)
    leg.append(i)

plot_data_plotly(Xs,Ys,leg)
