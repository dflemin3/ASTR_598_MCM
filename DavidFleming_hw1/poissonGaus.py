# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 21:37:17 2016

@author: dflemin3

hw1 ASTR 598 Monte Carlo Methods
"""

from __future__ import print_function, division
import numpy as np
from scipy.special import factorial
from scipy import integrate


def poisson(lam, m):
    return np.power(lam,m)*np.exp(-lam)/factorial(m)
    
# End function    
    
def CumulativePoisson(lam, m):
    
    m = int(np.floor(m)) # only use ints!
    
    if m < 0:
        print("m must be >= 0!")
        return -1
    
    tot = 0.0
    
    for i in range(0,m):
        tot += np.power(lam,i)/factorial(i)
    
    return tot*np.exp(-lam)
    
# End function
    
def gaussian(x,*args):
    mu = args[0]
    sigma = args[1]
    return np.exp(-np.power(x-mu,2.)/(2.*sigma*sigma))/(np.sqrt(2.*np.pi)*sigma)
    
# End function
    
def CumulativeGaussian(mu, sigma, y):
    if sigma <= 0:
        print("sigma must be >= 0!")
        return -1
    
    ret, err = integrate.quad(gaussian, -np.inf, y, args=(mu,sigma))
    return ret
    
# End Function