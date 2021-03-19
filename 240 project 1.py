#!/usr/bin/env python
# coding: utf-8

# # EECS 240 Project 1
# ## Student: Enyu Huang
# # 1
# Generate n=100 i.i.d Bernoulli Random Variables (RVs) with parameter P = 0:3
# <br>Generate some samples of the sum of these n RVs (You will need several sets of n Bernoulli RVs).
# <br>Draw the probability density function (PDF) of the sum.
# <br>Approximate the sum variable with a Gaussian RV and draw the corresponding PDF.
# <br>Produce the same figure for different n's and compare them.

# In[1]:


from scipy.stats import norm
from scipy.stats import bernoulli 
import matplotlib.pyplot as plt 
import numpy as np 
SampleTime=100000
p = 0.3
N = [10,100,1000]

Sum_Bernoulli = np.zeros(SampleTime) 
# Sum Of Bernoulli PDF
for n in N:
    sigma = (n*p*(1-p))**0.5
    mean = n*p
    for i in range(SampleTime):
        Sum_Bernoulli[i] = sum(bernoulli.rvs(p,size=n))
    plt.hist(Sum_Bernoulli,bins=40,color ='gray',density = True)
    x = np.arange(0, n) 
    y = norm.pdf(x, mean, sigma)
    plt.title('When N =%i' %n)
    plt.plot(x, y, color ='orange') 
    plt.xlabel('Gaussian')
    plt.ylabel('Probability')
    plt.show()


# # 2
# Generate n=100 i.i.d Poisson RVs with parameter  = 0:3
# <br>Generate some samples of the sum of these n RVs.
# <br>Draw the PDF of the sum.
# <br>Approximate the sum variable with a Gaussian RV and draw the corre-sponding PDF.
# <br>Produce the same figure for different n's and compare them.

# In[2]:


from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.stats import norm 
import matplotlib.pyplot as plt 
import numpy as np 
Sampletimes = 100000 
Lambda = 0.3
N = [10,100,1000]
Sum_Poisson = np.zeros(Sampletimes)
# PMF of the sum 
for n in N:
    for i in range(Sampletimes):
        Sum_Poisson[i] = sum(poisson.rvs(Lambda,size=n)) 
    plt.hist(Sum_Poisson,bins=40,color='pink',density=True)
    x = np.arange(0, n) 
    y = norm.pdf(x, n*Lambda, (n*Lambda)**0.5)
    plt.title('When N =%i' %n)
    plt.plot(x, y,color='blue')
    plt.xlabel('Gaussian')
    plt.ylabel('Probability')
    plt.show()


# # 3
# Generate CDF of a Gaussian RV with mean 2 and variance 3 from a uniformly distributed random variable in [0; 1].

# In[8]:


from scipy.stats import bernoulli 
from scipy.stats import poisson 
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
import numpy as np 
mean = 2 
var = 3
x = np.random.rand(1, 1000) # .rand generate from uniform distribution
y = norm.cdf(x, mean, var**(0.5))
plt.plot(y, x, 'o',color = 'gray')
plt.ylabel('Probability')
plt.xlabel('Value of GaussianRV') 
plt.show()


# # 4
# Generate two RVs; one binomial with parameters (6, 0.3), and the other one a
# Bernoulli with P=0.4. Then verify the law of large numbers by calculating the sample means.

# In[4]:


from scipy.stats import bernoulli 
from scipy.stats import poisson 
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np 

# Definition of Parameters 
binomial_mean = 6 
binomial_parameter = 0.3 
bernoulli_parameter = 0.4 
N = 10000
# Calculate Binomial sample mean
SampleMean_Binomial = [] 
for i in range(1, N):
    SampleMean_Binomial.append(sum(binom.rvs(binomial_mean, binomial_parameter,size=i))/i)
plt.plot(SampleMean_Binomial,label = 'SampleMean_Binomial')

# Calculate Bernoulli sample mean
SampleMean_Bernoulli = [] 
for i in range(1, N):
    SampleMean_Bernoulli.append(sum(bernoulli.rvs(bernoulli_parameter,size=i))/i)
plt.plot(SampleMean_Bernoulli,label = 'SampleMean_Bernoulli')
plt.legend()
plt.xlabel('Number of Sequences')
plt.ylabel('SampleMean') 
plt.show()


# # 5
# Estimate the mean of X^2, where X is a zero-mean, unit-variance Gaussian RV.

# In[5]:


from scipy.stats import bernoulli 
from scipy.stats import poisson 
from scipy.stats import binom
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np 
mean = 0 
var = 1 
N = 1000
Y = np.zeros(N)
YMean = []
for i in range(1,N):
    #get X as Gaussian RV
    X = norm.rvs(mean, var, size=N) 
for j in range(N):
    #compute the square of it
    Y[j] = X[j]**2
    YMean.append(np.mean(Y))
x = np.arange(0, N, 1) 
plt.ylim([0,1])
plt.plot(x,YMean)
plt.xlabel('Value of N')
plt.ylabel('Mean of X^2')
plt.show()

