#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import matplotlib.pyplot as plt
import math
from statsmodels.graphics.tsaplots import plot_acf


# In[2]:


#Problem 1.1
#when alpha = 0.3
N = 100
Y = numpy.zeros(N+1)
mean = 0
alpha = 0.3
std = math.sqrt(1-alpha**2)
Y[0] = numpy.random.normal(mean, std)
for i in range(N):
    X = numpy.random.normal(mean, std)
    Y[i+1] = alpha*Y[i] + X
print(Y.shape,Y)    
plt.title('Autoregressive Random Process')
plt.xlabel('when alpha = 0.3, N = 100')
plt.ylabel('Value')
plt.plot(Y)
plt.show()


# In[3]:


#when alpha = 0.95
N = 100
Y = numpy.zeros(N+1)
mean = 0
alpha = 0.95
std = math.sqrt(1-alpha**2)
Y[0] = numpy.random.normal(mean, std)
for i in range(N):
    X = numpy.random.normal(mean, std)
    Y[i+1] = alpha*Y[i] + X
print(Y.shape,Y)      
plt.title('Autoregressive Random Process')
plt.xlabel('when alpha = 0.95, N = 100')
plt.ylabel('Value')
plt.plot(Y)
plt.show()


# In[4]:


#Problem 1.2
#when alpha = 0.3
N = 100
Y = numpy.zeros(N+1)
mean = 0
alpha = 0.3
std = math.sqrt(1-alpha**2)
Y[0] = numpy.random.normal(mean, std)
for i in range(N):
    X = numpy.random.normal(mean, std)
    Y[i+1] = alpha*Y[i] + X
print(Y.shape,Y)  
plot_acf(Y)
plt.show()


# In[5]:


#when alpha = 0.95
N = 100
Y = numpy.zeros(N+1)
mean = 0
alpha = 0.95
std = math.sqrt(1-alpha**2)
Y[0] = numpy.random.normal(mean, std)
for i in range(N):
    X = numpy.random.normal(mean, std)
    Y[i+1] = alpha*Y[i] + X
print(Y.shape,Y)      
plot_acf(Y)
plt.show()


# In[10]:


#Problem 1.3
#when alpha = 0.3
from scipy import signal
N = 100
Y = numpy.zeros(N+1)
mean = 0
alpha = 0.3
std = math.sqrt(1-alpha**2)
Y[0] = numpy.random.normal(mean, std)
for i in range(N):
    X = numpy.random.normal(mean, std)
    Y[i+1] = alpha*Y[i] + X
sampling = 200
print(Y.shape,Y)      
freqs, psd = signal.welch(Y,sampling,scaling = 'density',nperseg = 101)
plt.semilogx(freqs, psd)
plt.show()


# In[7]:


#when alpha = 0.95
N = 100
Y = numpy.zeros(N+1)
mean = 0
alpha = 0.95
std = math.sqrt(1-alpha**2)
Y[0] = numpy.random.normal(mean, std)
for i in range(N):
    X = numpy.random.normal(mean, std)
    Y[i+1] = alpha*Y[i] + X
print(Y.shape,Y)      
sampling = 200
freqs, psd = signal.welch(Y,sampling,scaling = 'density',nperseg = 101)
plt.semilogx(freqs, psd)
plt.show()


# In[8]:


#Problem 2.1
import numpy
import matplotlib.pyplot as plt
import math
N = 100
pi = math.pi
X = numpy.zeros(N)
for i in range(N):
    theta = numpy.random.uniform(-pi, pi)
    X[i] = math.cos(0.2*pi*i+theta)
print(X)
plt.plot(X, 'o')
plt.show()

