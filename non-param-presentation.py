# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 12:22:43 2018

@author: DELL
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D


def kernel(a, b, covariance, lengthSpace):
    sqdist = covariance * np.exp(-cdist(a,b)/(2*(lengthSpace**2)))
    return sqdist


def f(x1, *args):
    # return the value of the objective at x
    x = x1.reshape(-1,2)
    Y = args[0]
    N = Y.shape[0]
    theta = args[1]
    #mean_Y = np.mean(Y, 0)
    S = np.dot(Y.T, Y)/N
    C = np.dot(x, x.T) + theta**2 * np.identity(x.shape[0])
    val = (N/2) * (np.log(np.linalg.det(C))) + np.trace(np.linalg.solve(C, S))
    return val


def dfx(x1,*args):
    # return the gradient of the objective at x
    x = x1.reshape(-1,2)
    Y = args[0]
    N = Y.shape[0]
    D = Y.shape[0]
    theta = args[1]
    C = np.dot(x, x.T) + theta**2 * np.identity(x.shape[0])
    #mean_Y = np.mean(Y, 0)
    S = np.dot(Y.T, Y)/N
    val = -N * (np.dot(np.linalg.solve(C, S), np.linalg.solve(C, x)) - D * np.linalg.solve(C, x))
    #print(np.asarray((val.flatten())))
    return np.asarray((val.flatten()))


def f_npp(x1, *args):
    # return the value of the objective at x
    x = x1.reshape(-1,2)
    Y = args[0]
    theta = args[1]
    covariance = args[2]
    lengthSpace = args[3]
    #D = Y.shape[0]
    K = kernel(x, x, covariance, lengthSpace) + theta**2 * np.identity(x.shape[0])
    val = (1/2)*((np.log(np.linalg.det(K))) + np.trace(np.dot(Y.T, np.linalg.solve(K, Y))))
    return val


def dfx_npp(x1,*args):
    # return the gradient of the objective at x
    x = x1.reshape(-1,2)
    Y = args[0]
    #D = Y.shape[0]
    theta = args[1]
    covariance = args[2]
    lengthSpace = args[3]
    K = kernel(x, x, covariance, lengthSpace) + theta**2 * np.identity(x.shape[0])
    dval = np.zeros(x.shape)
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            dK = np.zeros(K.shape)
            dK[i][j] = 1
            A = np.trace(np.dot(np.linalg.inv(K), dK))
            B = np.trace(np.dot(Y.T, np.linalg.solve(K, np.dot(dK, np.linalg.solve(K,Y)))))
            dval[i][j] = (K[i][j]*x[i][j]/(lengthSpace**2))*(A-B)
    return -np.asarray((dval.flatten()))


Y = np.load('non_param_data/data.npy')
Y_mean = np.mean(Y, 0)
Y_std = np.std(Y, 0)
Y = (Y - Y_mean) / Y_std
fig = plt.figure()
ax = Axes3D(fig)
j=100
for i in range(1, 41, 3):
    ax.scatter(Y[j][i], Y[j][i+1], Y[j][i+2])
plt.show()

x0 = np.random.normal(size=(Y.shape[1], 2))
x0 = x0.flatten()
theta = 0.1
args = (Y, theta)
x_star = opt.fmin_cg(f,x0,fprime=dfx, args=args)
W = x_star.reshape(-1,2)
C = np.dot(W, W.T) + theta**2 * np.identity(W.shape[0])
x_initial = np.dot(Y, np.dot(np.linalg.inv(C),W))
covariance = 0.1
lengthSpace = 0.1
args1 = (Y, theta, covariance, lengthSpace)
x_star = opt.fmin_cg(f_npp,x_initial,fprime=dfx_npp, args=args1)

X = x_star.reshape(-1,2)
plt.scatter(X[:,0], X[:,1])