# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:57:51 2016

@author: dflemin3
"""

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 20.0
mpl.rc('text', usetex='true') 

# Set the random seed
np.random.seed(int((time.time()%1)*1.0e8))

# Define the probability distribution function
def p(x):
    c = 1.0/5.01325654926 # Normalization coeff derived by integrating from -inf to inf
                          # using integrate.quad(p, -np.inf, np.inf)[0]
    return c*(np.exp(-np.power(x + 6.,2)/2.0) + np.exp(-np.power(x - 6.,2)/2.0))
    
    
############################################
#
# 1a: Plot the probability distribution p(x)
#
############################################
    
fig, ax = plt.subplots(figsize=(8,8))
x = np.linspace(-20,20,1000)
ax.plot(x,p(x), color="blue", lw=3, label=r"$P(x)$")

# Format plot
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$Normalized \ Probability$")
ax.set_title("1a")
ax.grid(True)
ax.legend(bbox_to_anchor=[1.7,1.0])
fig.savefig("1a.png")

############################################
#
# 1b: Implement the Metropolis algorithm
#
############################################

# Define the metropolis MCMC algorithm
def metropolis(x_int, P, N=10000, delta=1):
    
    # array to store result
    if N > 1.0e7:
        print("My laptop can't handle this!\n")
        return -1
    res = np.zeros(N)
    
    # Set x to be initial user-defined x
    x = x_int 
    
    # y = probability of initial x value
    y = P(x)
    
    # Do the following N times
    for i in range(0,N):
        # Get random value between -delta and delta
        s = np.random.uniform(low=-delta, high=delta)
        
        # The new trial value of x_new
        x_new = x + s
        y_new = P(x_new)
        
        # Metropolis algorithm core step
        if(y_new > y): # Accept new x_new as it's uphill!
            x = x_new
            y = y_new
        else: # Going downhill, sometimes accept step 
            # Get random value between 0 and 1
            r = np.random.uniform(low=0.0, high=1.0)
            
            if(y_new/y > r): # Accept x_new
                x = x_new
                y = y_new
            # end if
        # end if-else
        
        # Store result of step
        res[i] = x
        
    # end main for loop
    return res
# end function

############################################
#
# 1c: Run for delta=1, x_init = 5, N = 10000
#
############################################

delta = 1.0
x_int = 5
N = 10000

res = metropolis(x_int, p, N, delta)

fig, ax = plt.subplots(figsize=(8,8))

# Plot the histogram of my results
ax.hist(res, 50, normed=1, facecolor='green', alpha=0.75, label=r"$Metropolis \ Result$");

# Plot the true distribution
ax.plot(x,p(x), color="blue", lw=3, label=r"$P(x)$")

# Format
ax.set_xlabel(r"x")
ax.set_ylabel(r"$Probability$")
ax.set_title(r"1c")
ax.grid(True)
ax.legend(bbox_to_anchor=[1.7,1.0])
fig.savefig("1c.png")

############################################
#
# 1d: Run for delta=1, x_init = -5, N = 10000
#
############################################

delta = 1.0
x_int = -5
N = 10000

res = metropolis(x_int, p, N, delta)

fig, ax = plt.subplots(figsize=(8,8))

# Plot the histogram of my results
ax.hist(res, 50, normed=1, facecolor='green', alpha=0.75, label=r"$Metropolis \ Result$");

# Plot the true distribution
ax.plot(x,p(x), color="blue", lw=3, label=r"$P(x)$")

# Format
ax.set_xlabel(r"x")
ax.set_ylabel(r"$Probability$")
ax.set_title(r"1d")
ax.grid(True)
ax.legend(bbox_to_anchor=[1.7,1.0])
fig.savefig("1d.png")

############################################
#
# 1e: Run for delta=10, x_init = -5, N = 10000
#
############################################

delta = 10.0
x_int = -5
N = 10000

res = metropolis(x_int, p, N, delta)

fig, ax = plt.subplots(figsize=(8,8))

# Plot the histogram of my results
ax.hist(res, 50, normed=1, facecolor='green', alpha=0.75, label=r"$Metropolis \ Result$");

# Plot the true distribution
ax.plot(x,p(x), color="blue", lw=3, label=r"$P(x)$")

# Format
ax.set_xlabel(r"x")
ax.set_ylabel(r"$Probability$")
ax.set_title("1e")
ax.grid(True)
ax.legend(bbox_to_anchor=[1.7,1.0])
fig.savefig("1e.png")

############################################
#
# Extra: Run for delta=1.0, x_init = 0, N = 1000000
#
############################################

delta = 0.75
x_int = 0.0
N = 5000000

res = metropolis(x_int, p, N, delta)

fig, ax = plt.subplots(figsize=(8,8))

# Plot the histogram of my results
ax.hist(res, 50, normed=1, facecolor='green', alpha=0.75, label=r"$Metropolis \ Result$");

# Plot the true distribution
ax.plot(x,p(x), color="blue", lw=3, label=r"$P(x)$")

# Format
ax.set_xlabel(r"x")
ax.set_ylabel(r"$Probability$")
ax.set_title(r"Extra")
ax.grid(True)
ax.legend(bbox_to_anchor=[1.7,1.0])
fig.savefig("extra.png")
