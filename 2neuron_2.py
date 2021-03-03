#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:21:18 2021

@author: Krishna Soni, SRL_Rover
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:25:42 2021

@author: Krishna Soni
"""

import numpy as np
import random
import matplotlib.pyplot as plt
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
a = np.zeros((100,2), dtype=np.double)  #creation of the a matrix
z = np.zeros((100,2), dtype=np.double)         #creation of z matrix
#print(z)
w = np.array([random.random(), random.random()])  #initializing the weights with random values
b = np.array([random.random(), random.random()]) #initializing the baises with random values
X = np.zeros((100,1), dtype=np.double)
T = np.zeros_like(X)
E = np.zeros((100,1), dtype=np.double)

sumb0=0.0
sumw0=0.0 
sumb1=0.0 
sumw1=0.0
#a = [random.random(), random.random()]      #initializing the outputs with zeroes
#z = [random.random(), random.random()]      #initialising z with zeroes
#w = [random.random(), random.random()]  #initializing the weights with random values
#b = [random.random(), random.random()]  #initializing the baises with random values
for i in range(100):
    temp=random.random()
    X[i][0]=temp
    T[i][0]=temp**2

#plt.plot(X,T, 'ro')
#plt.axis([0,1,0,1])
#plt.show()
          
for j in range(1000):
    
    for i in range(100):
    #while E>0.00001*0.5*((T-a[1])**2) and j<10000:
       
        z[i][0] = w[0]*X[i][0] + b[0]   #finding the value of z1
        #a[0] = sigmoid(z[0])    #activating z1 with activation function
        a[i][0] = relu(z[i][0])
        
        z[i][1] = a[i][0]*w[1] + b[1] 
        #a[1] = sigmoid(z[1])
        a[i][1] = relu(z[i][1])
        
        E[i] = 0.5*((T[i][0]-a[i][1])**2)
        
        print("value of gradw1 in {} is {}".format(i,(-(T[i][0]-a[i][1])*(a[i][1]*(1-a[i][1]))*(a[i][0]))))
        sumw1 = sumw1 + (-(T[i][0]-a[i][1])*(a[i][1]*(1-a[i][1]))*(a[i][0]))
        sumb1 = sumb1 + (-(T[i][0]-a[i][1])*(a[i][1]*(1-a[i][1]))*(1))
        sumw0 = sumw0 + (-(T[i][0]-a[i][1])*(a[i][1]*(1-a[i][1]))*(1))
        sumb0 = sumb0 + (-(T[i][0]-a[i][1])*(a[i][1]*(1-a[i][1]))*(w[1])*(a[i][0]*(1-a[i][0]))*(1))
        print("value of symw1 in {} is {}".format(i,sumw1))
    
    w[1] = w[1] - tf*sumw1
    b[1] = b[1] - tf*sumb1
    w[0] = w[0] - tf*sumw0
    b[0] = b[0] - tf*sumb0
            
            #j= j+1
            
       # print("the final value of w in {} is {}".format(i,w))
       # print("the final value of b in {} is {}".format(i,b))
       # print("the final value of E is in {} {}".format(i,E))
       # print("the final value of input in {} is {}".format(i,X1))
       # print("the final value of true value in {} is {}".format(i,T))
       # print("the final value of output in {} is {}".format(i,a[1]))
      
        
    
print("the final value of w in is {}".format(w))
print("the final value of b in is {}".format(b))
end_time=datetime.now() 
print("Time in {} : {}".format(i,end_time-start_time))    
X2 = 0.5
T2 = X2**2
zcheck = [0, 0]
acheck = [0, 0]

zcheck[0] = w[0]*X2 + b[0]
acheck[0] = relu(zcheck[0])
zcheck[1] = acheck[0]*w[1] + b[1] 
acheck[1] = relu(zcheck[1])

print("check value of T2 is {}".format(T2))
print("The value predicted by network is {}".format(acheck[1]))