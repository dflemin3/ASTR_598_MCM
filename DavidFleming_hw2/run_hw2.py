# David Fleming hw2

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import factorial
import time

#Typical plot parameters that make for pretty plots
plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['font.size'] = 15.0

# Set the random seed
seed=time.time()
print("Seed: %lf" % seed)

# Define functions to compute poisson and gaussian probabilities
def poisson(lam, m):                                                                                                              
    return np.power(lam,m)*np.exp(-lam)/factorial(m)

def gaussian(x,*args):
    mu = args[0]
    sigma = args[1]                                                                                                                
    return np.exp(-np.power(x-mu,2.)/(2.*sigma*sigma))/(np.sqrt(2.*np.pi)*sigma)

# QUESTION 1a

# Generate 10^6 random deviates from the poisson distribution
N = 100000
N_bins = 50
lam = 10.0
binwidth = 1

data = np.random.poisson(lam,N)

# Plot
fig, ax = plt.subplots(figsize=(8,8))

# Plot histogram of 10^6 random deviates from poisson distribution with lambda = 10
n, bins, patches = ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), 
                           normed=0, facecolor='blue', alpha=0.75, label='10^6 Random',
                          align='left')
mid_bins = np.floor(0.5*(bins[1:]+bins[:-1]))
y_fit = N*poisson(lam,mid_bins)

# Plot NP(n)
ax.plot(mid_bins,y_fit,"o-",label="NP(n)", lw=3)

# Format plot
ax.set_ylabel("Counts")
ax.set_xlabel("n")
ax.legend(loc="upper right")
ax.grid(True)

fig.savefig("fig1a.png")

# QUESTION 1b

# Generate 10^2 random deviates from the poisson distribution
N = 100
N_bins = 50
lam = 10.0

data = np.random.poisson(lam,N)

#Plot
fig, ax = plt.subplots(figsize=(8,8))

# Plot histogram of 10^6 random deviates from poisson distribution with lambda = 10
n, bins, patches = ax.hist(data, bins=range(min(data), max(data) + binwidth, binwidth), 
                           normed=0, facecolor='blue', alpha=0.75, label='10^2 Random',
                          align='left')
mid_bins = np.floor(0.5*(bins[1:]+bins[:-1]))

# Compute the NP(n) fit
y_fit = N*poisson(lam,mid_bins)

# Plot NP(n)
ax.plot(mid_bins,y_fit,"o-",label="NP(n)", lw=3)

# Format plot
ax.set_ylabel("Counts")
ax.set_xlabel("n")
ax.legend(loc="upper right")
ax.grid(True)

fig.savefig("fig1b.png")

#QUESTION 2a

mu = 0.0
sigma = 10.0

# Generate 10^6 random deviates from the poisson distribution
N = 100000
N_bins = 50
binwidth = 1

data = np.random.normal(mu, sigma, N)

# Plot
fig, ax = plt.subplots(figsize=(8,8))

# Plot histogram of 10^6 random deviates from poisson distribution with lambda = 10
n, bins, patches = ax.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth), 
                           normed=0, facecolor='blue', alpha=0.75, label='10^6 Random',
                          align='left')
mid_bins = 0.5*(bins[1:]+bins[:-1])

# Compute the NP(n) fit
y_fit = N*gaussian(mid_bins,mu,sigma)

# Plot NP(n)
ax.plot(mid_bins,y_fit,"o-",label="NP(n)", lw=3)

# Format plot
ax.set_ylabel("Counts")
ax.set_xlabel("n")
ax.legend(loc="upper left")
ax.grid(True)

fig.savefig("fig2a.png")

# QUESTION 2b

# Generate 10^2 random deviates from the poisson distribution
N = 100
N_bins = 50
binwidth = 1

data = np.random.normal(mu, sigma, N)

# Plot
fig, ax = plt.subplots(figsize=(8,8))

# Plot histogram of 10^6 random deviates from poisson distribution with lambda = 10
n, bins, patches = ax.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth), 
                           normed=0, facecolor='blue', alpha=0.75, label='10^2 Random',
                          align='left')
mid_bins = 0.5*(bins[1:]+bins[:-1])

# Compute the NP(n) fit
y_fit = N*gaussian(mid_bins,mu,sigma)

# Plot NP(n)
ax.plot(mid_bins,y_fit,"o-",label="NP(n)", lw=3)

# Format plot
ax.set_ylabel("Counts")
ax.set_xlabel("n")
ax.legend(loc="upper left")
ax.grid(True)

fig.savefig("fig2b.png")
