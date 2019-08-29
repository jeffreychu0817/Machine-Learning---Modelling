# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 08:11:47 2018

@author: DELL
"""
import numpy as np
#from math import pi
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 
import math

def kernel(a, b, covariance, lengthSpace):
    sqdist = covariance * np.exp(-cdist(a,b)/(2*(lengthSpace**2)))
    return sqdist


def GaussianProcess(Xtest, X, y, covariance, lengthSpace, gaussian_covariance):
    K = kernel(X, X, covariance, lengthSpace)
    K_ = kernel(Xtest, Xtest, covariance, lengthSpace)
    K_mix = kernel(Xtest, X, covariance, lengthSpace)
    mean = np.dot(K_mix, np.linalg.solve(K+gaussian_covariance*np.eye(K.shape[0]), y))
    variance = np.diag(K_ - np.dot(K_mix, np.linalg.solve(K+gaussian_covariance*np.eye(K.shape[0]), K_mix.T)))
    return(mean.squeeze(), variance.squeeze())


num_training = 10
num_test = 50
variance = 0.01**2
#x_test = np.array([-1]).reshape(-1,1)
x_test = np.linspace(-5, 5, num_test).reshape(-1,1)
#x_training = np.arange(0, num_training * pi/5 - pi/180, pi/5).reshape(-1,1)
x_training = np.linspace(-3, 3, num_training).reshape(-1,1)
y_training = (np.sin(x_training) + np.random.normal(0, math.sqrt(variance), num_training).reshape(-1,1))
K = kernel(x_test, x_test, 1, 0.1)
L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
f_prior = np.dot(L, np.random.normal(size=(L.shape[0],1)))
plt.legend('l=0.1')

#fig, ax = plt.subplots()
#
#K = kernel(x_test, x_test, 1, 0.1)
#L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
#f_prior = np.dot(L, np.random.normal(size=(L.shape[0],1)))
#plt.legend('l=0.1')
#ax.plot(x_test, f_prior, label='l=0.1')
#
#K = kernel(x_test, x_test, 1, 1)
#L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
#f_prior = np.dot(L, np.random.normal(size=(L.shape[0],1)))
#ax.plot(x_test, f_prior, label='l=1')
#
#K = kernel(x_test, x_test, 1, 10)
#L = np.linalg.cholesky(K + 1e-6*np.eye(K.shape[0]))
#f_prior = np.dot(L, np.random.normal(size=(L.shape[0],1)))
#ax.plot(x_test, f_prior, label='l=10')
#
#legend = ax.legend(loc='upper right', shadow=True, fontsize='x-large')
#
#plt.show()

mean, variance = GaussianProcess(x_test, x_training, y_training, 1, 1, variance)
stdv = np.sqrt(variance)
plt.gca().fill_between(x_test.flat, mean-2*variance, mean+2*variance, color="#dddddd")
plt.scatter(x_test, mean, c='r')
plt.scatter(x_training, y_training)       
plt.plot()