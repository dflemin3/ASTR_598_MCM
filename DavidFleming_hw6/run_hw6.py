# -*- coding: utf-8 -*-
"""
David Fleming
hw6 script
"""

from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (8,8)
mpl.rcParams['font.size'] = 30.0

from matplotlib import rc
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

# Set the random seed
np.random.seed(int((time.time()%1)*1.0e8))

##############################################
#
# 1a: Run ensemble of 2D lattice random walks
#
###############################################

# Define function to run 1 random walk that
# takes N steps through a 2d lattice
def random_walk_2d(N):
    # For N steps, do a 2d random walk on some lattice in xy space

    # Initially start at the origin
    x = 0.0
    y = 0.0

    for i in range(0,N):

        # Get random numbers for x, y motion
        r = np.random.uniform(0,1)

        if(r < 0.25): # Move left
            x = x - 1.0
        elif(r >= 0.25 and r < 0.5): # Move up
            y = y + 1.0
        elif(r >= 0.5 and r < 0.75): # Move right
            x = x + 1.0
        else: # Move down
            y = y - 1.0

    # Return squared distance from the origin
    # and the final coords
    return x**2 + y**2, x, y

# end function

# Define an random walk ensemble function that
# runs num random walks each of N steps
def random_walk_2d_ensemble(num, N):
    # Run num random walk runs each of N length

    # Allocate arrays to store results of random walk runs
    res = np.zeros(num)

    for i in range(0,num):
        res[i] = random_walk_2d(N)[0]

    # Return mean, std
    return np.mean(res), np.std(res)

# end function

# Run ensemble of random walkers for various step
# numbers with 100 runs for each step

N = np.array([100,1000,10000,100000])
num = 100

means = np.zeros(len(N))
stds = np.zeros(len(N))

for i in range(0,len(N)):
    means[i], stds[i] = random_walk_2d_ensemble(num, N[i])


##############################################
#
# 1b: Plot <s^2> vs N using sigma as the error
#
###############################################

fig, ax = plt.subplots(figsize=(10,10))

# Plot theoretical value
expected = N

ax.errorbar(N,means,yerr=stds,fmt="o",label=r"Observed $s^2$")
ax.plot(N,expected,"o-",label=r"Expected s^2 \approx N")

ax.set_xlabel("N")
ax.set_ylabel(r"$<s^2>$")

ax.set_xlim(N.min()-0.1*N.min(),N.max()+0.1*N.max())
ax.set_xlim(means.min()-0.1*means.min(),means.max()+0.1*means.max())

ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(loc="upper left", fontsize=20)

fig.tight_layout()
fig.savefig("hw6_1b.png")