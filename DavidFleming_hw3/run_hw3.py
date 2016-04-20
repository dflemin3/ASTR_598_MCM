# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:54:15 2016

Script to answer questions in hw3 from 
ASTR 598 Monte Carlo Methods, Spring 2016

@author: dflemin3
"""

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
mpl.use("Agg") # Set for headless hyak cluster
import matplotlib.pyplot as plt
from scipy.special import factorial
import time

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 15.0
mpl.rc('text', usetex='true') 

# Seed RNG using subsecond variations incase this gets run a lot
np.random.seed(int((time.time()%1)*1.0e8))

#############################
## Define useful functions ##
#############################

# Define functions to compute poisson and gaussian probabilities
def poisson(m,*args):
    lam = float(args[0])

    # Make sure input is int!    
    if type(m) == float:
        n = int(m)
    elif type(m) == np.ndarray:
        n = m.astype(int)
    else:
        n = m
        
    # Make lambda a float to prevent int overflow!
    return np.power(lam,n)*np.exp(-lam)/factorial(n)
# end function

def gaussian(x,*args):
    mu = args[0]
    sigma = args[1]                                                                                                                
    return np.exp(-np.power(x-mu,2.)/(2.*sigma*sigma))/(np.sqrt(2.*np.pi)*sigma)
# end function

def find_YMAX_poisson(lam):
    # Find the int that gives the max values
    # of the poisson distribution P(n), YMAX
    
    val = 0.0
    m = 0

    while True:
        tmp = poisson(m,lam)
        
        # If increasing, save value
        if tmp > val:
            val = tmp
            m_max = m
        else:
            break
    
        m = m + 1
    # end while
    
    return m_max, val
# end function
    
def find_XMAX_poisson(lam,m_max,eps=1.0e-8,MAX=50):
    # Given the index for the maximum of the poisson function,
    # find the index at which the Poisson distribution decays
    # to near 0, XMAX

    m = m_max
    count = 0

    # Start from peak and move rightward until distribution goes to 0
    while count < MAX:
        tmp = poisson(m,lam)
        
        # poisson is 0 enough
        if tmp < eps:
            return m
        
        m = m + 1
        count = count + 1
    
    return -1.0        
# end function
   
#####################
#
# Problem 1a
#   
#####################   
   
# Return n value drawn from the poisson
# 1 dimensional case for the poisson distribution
def nextPoisson(n,lam,MAX_ATTEMPTS=40):
      
    # Default inits    
    m_max, YMAX = find_YMAX_poisson(lam) 
    XMAX = find_XMAX_poisson(lam,m_max)    
    
    ret = []    
    
    for i in range(0,n):
        count = 0
        while(count < MAX_ATTEMPTS):
            x0 = np.random.uniform(low=0.0,high=XMAX)
            y0 = np.random.uniform(low=0.0,high=YMAX)
            
            # Increment count
            count += 1
            
            # Does it fall under the distribution
            if y0 < poisson(x0,lam):
                ret.append(x0)
                break
                
            # too many attempts!
            if count >= MAX_ATTEMPTS:
                print("Error: Too many attempts while calling rejection_sampling!")
                ret.append(0.0)
                break
        #end while
    
    return np.asarray(ret)
# end function
 
#####################
#
# Problem 2a
#   
#####################
           
# Returns n random deviates drawn from a Gaussian
# using the Box-Mueller Transform
# where args = (mu,sigma)
def nextGaussian(n,*args):

    # Extra args
    mu = args[0]
    sigma = args[1]
    
    # x1, x2 are uniform randoms from [0,1]
    # Draw 2 since BM transform returns 2 distributed according to normal distribution
    x1 = np.random.uniform(size=int(n/2))
    x2 = np.random.uniform(size=int(n/2))
    U1 = 1.0 - x1
    U2 = 1.0 - x2
    
    z0 = np.sqrt(-2.0 * np.log(U1)) * np.cos(2.0*np.pi * U2)
    z1 = np.sqrt(-2.0 * np.log(U1)) * np.sin(2.0*np.pi * U2)
    
    # Returns vector of normally distributed data!
    return np.concatenate([z0 * sigma + mu,z1 * sigma + mu])
# end function
    
#####################################   
### Run the rest of the homework  ###
#####################################

#####################
#
# Problem 1b
#   
#####################

num= 1000000
lam = 10.0
binwidth = 1

# Generate the data
data = nextPoisson(num,lam)

# Plot
fig, ax = plt.subplots(figsize=(8,8))

# Plot histogram of 10^6 random deviates from poisson distribution with lambda = 10
n, bins, patches = ax.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth), 
                           normed=0, facecolor='blue', alpha=0.75, label="10$^6$ Random",
                          align='left')
# Overplot fit
mid_bins = np.floor(0.5*(bins[1:]+bins[:-1]))
y_fit = num*poisson(mid_bins,lam)

# Plot NP(n)
ax.plot(mid_bins,y_fit,"o-",label="NP(n)", lw=3)

# Format plot
ax.set_ylabel("Counts")
ax.set_xlabel("n")
ax.legend(loc="upper right")
ax.grid(True)

fig.savefig("DavidFleming_hw3_1b.png")

#####################
#
# Problem 2b
#   
#####################

mu = 0.0
sig = 10.0

data = nextGaussian(num,mu,sig)

# Plot
fig, ax = plt.subplots(figsize=(8,8))

# Plot histogram of 10^6 random deviates from poisson distribution with lambda = 10
n, bins, patches = ax.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth), 
                           normed=0, facecolor='blue', alpha=0.75, label="10$^6$ Random",
                          align='left')
# Overplot fit
mid_bins = np.floor(0.5*(bins[1:]+bins[:-1]))
y_fit = num*gaussian(mid_bins,mu,sig)

# Plot NP(n)
ax.plot(mid_bins,y_fit,"o-",label="NP(x)", lw=3)

# Format plot
ax.set_ylabel("Counts")
ax.set_xlabel("n")
ax.legend(loc="upper right")
ax.grid(True)

fig.savefig("DavidFleming_hw3_2b.png")