#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:25:42 2021

@author: Krishna Soni
"""

import numpy as np
import random
from datetime import datetime

start_time = datetime.now()
random.seed(a=None, version=2)  #initialise the seed to the system time


def sigmoid(x):     #function to calculate the sigmoid
    return 1/(1+np.exp(-x))

def relu(x):
    if x>0:
        return x
    else:
        return 0
    
tf = 0.2    #training factor
a = [random.random(), random.random()]      #initializing the outputs with zeroes
z = [random.random(), random.random()]      #initialising z with zeroes
w = [random.random(), random.random()]  #initializing the weights with random values
b = [random.random(), random.random()]  #initializing the baises with random values

for i in range (100000):
    X1 = random.random()    #input to neural network
    T = sigmoid(X1)    #the true value of output
    E = 1.0
    j = 0
    
    
    
    for j in range(1000):
    #while E>0.00001*0.5*((T-a[1])**2) and j<10000:
       
        z[0] = w[0]*X1 + b[0]   #finding the value of z1
        #a[0] = sigmoid(z[0])    #activating z1 with activation function
        a[0] = relu(z[0])
        
        z[1] = a[0]*w[1] + b[1] 
        #a[1] = sigmoid(z[1])
        a[1] = relu(z[1])
        
        E = 0.5*((T-a[1])**2)
        
        #print("value of E in {} is {}".format(i,E))
        
        w[1] = w[1] - tf*(-(T-a[1])*(a[1]*(1-a[1]))*(a[0]))
        b[1] = b[1] - tf*(-(T-a[1])*(a[1]*(1-a[1]))*(1))
        w[0] = w[0] - tf*(-(T-a[1])*(a[1]*(1-a[1]))*(w[1])*(a[0]*(1-a[0]))*(X1))
        b[0] = b[0] - tf*(-(T-a[1])*(a[1]*(1-a[1]))*(w[1])*(a[0]*(1-a[0]))*(1))
        
        #j= j+1
        
    print("the final value of w in {} is {}".format(i,w))
    print("the final value of b in {} is {}".format(i,b))
    print("the final value of E is in {} {}".format(i,E))
    print("the final value of input in {} is {}".format(i,X1))
    print("the final value of true value in {} is {}".format(i,T))
    print("the final value of output in {} is {}".format(i,a[1]))
    
    
end_time=datetime.now() 
print("Time in {} : {}".format(i,end_time-start_time))    
X2 = 0.5
T2 = sigmoid(X2)
z[0] = w[0]*X1 + b[0]
a[0] = relu(z[0])
z[1] = a[0]*w[1] + b[1] 
a[1] = relu(z[1])

print("check value of T2 is {}".format(T2))
print("The value predicted by network is {}".format(a[1]))