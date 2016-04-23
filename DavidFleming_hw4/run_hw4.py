# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:50:36 2016

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
mpl.rcParams['font.size'] = 15.0

# Set the seed
np.random.seed(int((time.time()%1)*1.0e8))

def volume4Dsphere(radius, N):
    """
    Perform a monte carlo integration to compute the radius of a 4d "sphere"
    using N samples
    """
    
    # Sanity checks
    if radius <= 0.0:
        print("ERROR: Radius must be greater than 0.")
        return -1
    if N <= 0:
        print("ERROR: Number of integration points must be greater than 0")
        return -1
    
    # Draw 4 random deviates from 0 to radius
    # These sample 1/16 of sphere volume... like the equivalent of the 1st quadrant of the sphere
    # Volume of "1st quadrant" goes as 16*radius^4 (like )
    x = np.random.uniform(low=0.0, high=radius, size=N)
    y = np.random.uniform(low=0.0, high=radius, size=N)
    z = np.random.uniform(low=0.0, high=radius, size=N)
    w = np.random.uniform(low=0.0, high=radius, size=N)

    return 16.0*np.power(radius,4.0)*np.sum(x**2 + y**2 + z**2 + w**2 <= radius**2.0)/N
# end function    
    
def monteCarloIntegrate(radius,N,num=10):
    res = []    
    for i in range(0,num):
        res.append(volume4Dsphere(radius, N))
    return np.mean(res), np.std(res)
# end function
    
radius = 2.0
N = np.asarray([int(1e6),int(1e7),int(1e8),int(1e9)])

mean = []
std = []
radius = 1.0
for i in range(0,len(N)):
    tmp_mean, tmp_std = monteCarloIntegrate(radius,N[i])
    print("N: ",N[i])
    print("Mean: ",tmp_mean)
    print("Std: ",tmp_std)
    mean.append(tmp_mean)
    std.append(tmp_std)
    
fig, ax = plt.subplots(figsize=(8,8))

ax.errorbar(N,mean,yerr=std,label=r"Monte Carlo Integration",
            fmt='o')
ax.set_xlabel("logN")
ax.set_ylabel("4-Volume")
ax.set_xlim(N.min()/10.0,N.max()*10.0)
ax.set_xscale("log")
ax.grid(True)


fig.savefig("DavidFleming_hw4d.png")
