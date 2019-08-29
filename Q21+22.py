# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 16:21:50 2018

@author: DELL
"""
import numpy as np
from math import pi
import scipy.optimize as opt
import matplotlib.pyplot as plt


def f(x1, *args):
    # return the value of the objective at x
    x = x1.reshape(10,2)
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
    x = x1.reshape(10,2)
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


X = np.linspace(0, 4*pi, 100).reshape(-1, 1)
f_non_linear = np.hstack((np.multiply(X, np.sin(X)), np.multiply(X, np.cos(X))))
X1 = f_non_linear
A = np.random.normal(size=(10,2))
Y = np.dot(f_non_linear, A.T)
#Y += np.random.normal(scale=1., size=Y.shape)
#plt.subplot(2, 1, 1)
#plt.scatter(X1[:,0], X1[:,1])

x0 = np.random.normal(size=(Y.shape[1], X1.shape[1]))
x0 = x0.flatten()
theta = 0.01
args = (Y, theta)
x_star = opt.fmin_cg(f,x0,fprime=dfx, args=args)
W = x_star.reshape(10,2)
C = np.dot(W, W.T) + theta**2 * np.identity(W.shape[0])
X_restore = np.dot(Y, np.dot(np.linalg.inv(C),W))
plt.axis('equal')
plt.subplot(2, 1, 1)
plt.scatter(X_restore[:,0], X_restore[:,1])

plt.subplot(2, 1, 2)
w_random = np.random.normal(size=(Y.shape[1], X1.shape[1]))
C_random = np.dot(w_random, w_random.T) + theta**2 * np.identity(W.shape[0])
X_random = np.dot(Y, np.dot(np.linalg.inv(C_random),w_random))
plt.scatter(X_random[:,0], X_random[:,1])