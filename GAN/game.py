import numpy as np
import sys
import os

sys.path.append("/home/gabriel/projects/tcl_code")

from utility import *

def adversarial_update(x,y,lr):
    x = x - lr*y
    y = y - lr*(-x)
    return x,y

def make_noise_update(noise=0.1):
    def noise_update(x,y,lr):
        x,y = adversarial_update(x,y,lr)
        x = x + noise*np.random.randn()
        y = y + noise*np.random.randn()
        return x,y
    return noise_update

def make_ascent_update(prob=0.1):
    def ascent_update(x,y,lr):
        xn,yn = adversarial_update(x,y,lr)
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

#def make_random_move_update(prob=0.1, max_steps=5):
#    def random_move_update(x,y,lr):


def sim_game(n, update, lr=0.1):
    x = 4*(2*np.random.rand()-1)
    y = 4*(2*np.random.rand()-1)

    xs = []
    ys = []
    xs.append(x)
    ys.append(y)
    for i in range(0,n):

        x,y = update(x,y,lr)

        xs.append(x)
        ys.append(y)

    return xs,ys

noise_update = make_noise_update(0.2)
ascent_update = make_ascent_update(0.3)

Ngames = 4
steps = 1000
Xs = []
Ys = []
leg = []
for i in range(0,Ngames):
    #xl,yl = sim_game(steps, adversarial_update, lr=0.1)
    xl,yl = sim_game(steps, noise_update, lr=0.1)
    #xl,yl = sim_game(steps, ascent_update, lr=0.1)
    Xs.append(xl)
    Ys.append(yl)
    leg.append(i)

plot_data_plotly(Xs,Ys,leg)
