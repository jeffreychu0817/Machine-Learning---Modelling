# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:20:05 2018

@author: DELL
"""

import numpy as np
from math import pi
import math
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def DrawGaussian(mean, variance):
    x_draw, y_draw = np.mgrid[-2:2:.1, -2:2:.1]
    pos = np.empty(x_draw.shape + (2,))
    pos[:, :, 0] = x_draw; pos[:, :, 1] = y_draw
    rv = multivariate_normal([mean[0][0], mean[1][0]], variance)
    plt.contourf(x_draw, y_draw, rv.pdf(pos))
    plt.plot(-1.3,0.5,'ro')
    plt.show()

def calculatePosterior(variance, x, y, sigma):
    var = np.linalg.inv(np.dot(np.transpose(x),x)/variance + np.linalg.inv(sigma))
    mean = np.dot(var, np.dot(np.transpose(x), y))/variance
    return var, mean


def addPoint(variance, mean, x, y):
    new_var = variance + np.dot(np.transpose(x),x)
    new_mean = mean +  np.dot(new_var, np.dot(np.transpose(x), y))/variance
    return new_var, new_mean

#f = np.random.multivariate_normal(mu, K)
#d = cdist(x1,x2)
#E = np.exp(D)
#print(np.linalg.inv(np.array([[2,0],[0,2]])))
X_data = np.arange(-1, 1.01, 0.01).reshape(-1,1)
X = np.hstack((np.ones((X_data.shape[0], 1)), X_data))
w = np.array([-1.3, 0.5]).reshape(-1, 1)
variance = np.power(0.03,2)
y = X.dot(w) + np.random.normal(0, math.sqrt(variance), X.shape[0]).reshape(-1,1)
variance = np.power(0.2,2)
#y = X.dot(w)
plt.plot(X_data, y)
plt.show()
prior_variance = np.array([[1/2, 0], [0, 1/2]])
prior_mean = np.array([0,0]).reshape(-1,1)
DrawGaussian(prior_mean, prior_variance)

x1 = np.array([1,X_data[10]]).reshape(1, 2)
y1 = np.array(y[10]).reshape(1,1)

x2 = np.array([1,X_data[20]]).reshape(1, 2)
y2 = np.array(y[20]).reshape(1,1)

x3 = np.array([1,X_data[20]]).reshape(1, 2)
y3 = np.array(y[20]).reshape(1,1)

v1, m1 = calculatePosterior(variance, x1, y1, prior_variance)
v2, m2 = calculatePosterior(variance, np.vstack((x1, x2)), np.vstack((y1, y2)), prior_variance)
v3, m3 = calculatePosterior(variance, np.vstack((x1, x2, x3)), np.vstack((y1, y2, y3)), prior_variance)



rd = random.sample(range(0, 199), 7)
v4, m4 = calculatePosterior(variance, np.vstack((x1, x2, x3, X[rd])), np.vstack((y1, y2, y3, y[rd])), prior_variance)
#plt.plot(X[rd][:,1], y[rd], 'rx')
#plt.show()
DrawGaussian(m1, v1)
DrawGaussian(m2, v2)
DrawGaussian(m3, v3)
DrawGaussian(m4, v4)
#mean1 = 3*(3*x1*x1 + )


