# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:41:37 2016

@author: dflemin3
"""

from __future__ import print_function, division
import poissonGaus as pg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#Typical plot parameters that make for pretty plots
plt.rcParams['figure.figsize'] = (8,8)
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rcParams['font.size'] = 20.0

# Problem 1a
print("Plotting poisson PDF for various lambdas...")
lam = [0.5, 2, 4, 8]
n = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

fig, ax = plt.subplots()

for val in lam:
    ax.plot(n,pg.poisson(val,n), '-o',label=r'$\lambda = %.1lf$' % val, lw=3)
    
ax.set_xlabel(r"n")
ax.set_ylabel(r"Probability")
ax.grid(True)
ax.legend()
#fig.savefig("DavidFleming1a.pdf")

print("If lambda = 0.1, what's CumulativePoisson(lambda,int(100*lambda))?")
print("The answer is: %lf." % pg.CumulativePoisson(0.1,100*0.1))

# Gaussian plots
print("Plotting Gaussian PDF for various sigmas...")
fig, ax = plt.subplots(figsize=(8,8))

mu = 0.0
sigma = [0.5, 2, 4, 8]
x = np.linspace(-20,20,250)

for sig in sigma:
    ax.plot(x,pg.gaussian(x,*(mu,sig)),label=r'$\mu = 0$, $\sigma = $%.1lf' % sig, lw=3)
    
ax.set_ylabel(r"Probability")
ax.set_xlabel(r"x")
ax.grid(True)
ax.legend()
#fig.savefig("DavidFleming2a.pdf")
