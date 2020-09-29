### kodingan tentang numpy

import os
import pandas as pd
import numpy as np
import math,datetime
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


## buat numpy array
np1dimensi = np.array([1,2,3,4],dtype = 'float32')#pemakaian dtype opsional
np2dimensi = np.array([(1,2,3,4),(5,6,7,8)])
np3dimensi = np.array([[(1,2,3,4),(5,6,7,8)],[(9,10,11,12),(13,14,15,16)]])
print(np1dimensi.ndim,np2dimensi.ndim,np3dimensi.ndim)

##karakteristik suatu numpy array
cc = np.arange(40).reshape(5,8)
print(cc.ndim)
print(cc.shape)
print(cc.size)
print(cc.dtype)
print(type(c))


##reshaping numpy array
c = np.arange(30).reshape(5,6)

##menginsert kesuatu numpy array
c = np.arange(30).reshape(5,6)
d = np.insert(c,1,np.arange(80,85),axis=1)
e = np.insert(c,3,np.arange(71,77),axis=0)

##boleh dicoba nih
coba = np.arange(15).reshape(3,5)
t = np.zeros((1,coba.shape[1]))
for i in np.arange(coba.shape[0]):
    t = np.insert(t,i+1,coba[i:i+1],axis=0)
f = np.delete(t,0,axis=0)


##boleh dicoba lagi 
a = np.arange(10,90).reshape(10,8)
b = np.zeros((1,a.shape[1]))
for i in np.arange(len(a)):
	b = np.insert(b,i+1,a[i:i+1],axis = 0)#b beda dengan c
b = np.delete(b,0,axis = 0)
c = np.zeros((a.shape[0],1))
for x in np.arange(a.shape[1]):
	c = np.insert(c,[x+1],a[:,x:x+1],axis = 1)#c beda dengan b
c = np.delete(c,0,axis = 1)
print(a)
print(b)
print(c)
