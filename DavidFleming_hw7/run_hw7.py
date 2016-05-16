# -*- coding: utf-8 -*-
"""
David Fleming 2016

ASTR 598 Monte Carlo Methods Homework 7 Solution:
Implementation of a self-avoiding random walk

"""

# Imports
from __future__ import division, print_function

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time

#Typical plot parameters that make for pretty plots
mpl.rcParams['figure.figsize'] = (9,8)
mpl.rcParams['font.size'] = 25.0

from matplotlib import rc
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)

# Set the random seed
np.random.seed(int((time.time()%1)*1.0e8))


########################################
#
# 1a) Self-avoiding random walk
#
########################################

# Self-avoiding random walk implementation
# Returns 0 distance traveled if walk is rejected
# Rejection occurs when chain attempts to go where
# it already exists
def self_avoiding_random_walk(steps):
    bFirstStep = True

    history = [(0,0,0)] # Contains history of steps (tuples of (x,y,z)) we wish to avoid

    i = 0 # step counter

    # Directions for 3d lattice
    new = [xx for xx in range(1,7)]

    # Initial position
    x = 0
    y = 0
    z = 0
    past = 0

    #Run the random walk starting from the origin, (0,0,0)
    for i in range(0,steps):

        if bFirstStep: # Can go any of 6 ways initially
            prob = 1./6. # Probability of moving a given direction

            # Get random number
            r = np.random.uniform(0.0,1.0)

            # Pick a direction
            if r <= prob:
                x = x + 1 # Go right
                past = 1
            elif r > prob and r <= 2.0*prob:
                x = x - 1 # Go left
                past = 2
            elif r > 2.0*prob and r <= 3.0*prob:
                y = y + 1 # Go top
                past = 3
            elif r > 3.0*prob and r <= 4.0*prob:
                y = y - 1 # Go bottom
                past = 4
            elif r > 4.0*prob and r < 5.0*prob:
                z = z + 1 # Go up
                past = 5
            else:
                z = z - 1 # Go down
                past = 6

            # Add this step to step history
            history.append((x,y,z))

            # Not the first step anymore
            bFirstStep = False

        # Not first step, can only go 1 of 5 directions
        else:
            # Exclude the past direction
            new.remove(past)

            old_past = past # Cache past step

            prob = 1./5. # Probability to move in a given direction

            # Get random number
            r = np.random.uniform(0.0,1.0)

            # Loop over directions
            for i in range(0,len(new)):
                if r > i*prob and r <= (i+1)*prob:
                    next_step = new[i]
                    break

            # Now step, save which direction you step
            if next_step == 1:
                x = x + 1 # Go right
                past = 1
            elif next_step == 2:
                x = x - 1 # Go left
                past = 2
            elif next_step == 3:
                y = y + 1 # Go top
                past = 3
            elif next_step == 4:
                y = y - 1 # Go bottom
                past = 4
            elif next_step == 5:
                z = z + 1 # Go up
                past = 5
            else:
                z = z - 1 # Go down
                past = 6

            # Add old_past back to direction list
            new.append(old_past)

            # Now check if we've been here before
            current_position = (x,y,z)

            # Been there, reject this iteration
            if current_position in history:
                return [(0,0,0)], 0.0
            # Haven't been there, add step and repeat
            else:
                # Add this step to your history
                history.append((x,y,z))

    # Simulation over: return final position and squared distance from origin
    return history[-1], (history[-1][0]**2 + history[-1][1]**2 + history[-1][2]**2)
# end function

# Define a function to run an ensembled of self-avoiding random walks
# N: number of random walks to attempt.  May not get 10000 distances s2
# out because we only want walks that are accepted for our statistics
def self_avoiding_ensemble(steps,N=10000):
    s2 = []
    acceptance = 0.0

    for i in range(0,N):
        tmps2 = self_avoiding_random_walk(steps)[1] # only save square distance from origin

        # Did it actually go anywhere? If so, save this
        if tmps2 > 0.0:
            acceptance = acceptance + 1.0
            s2.append(tmps2)

    # Return mean, std
    s2 = np.array(s2)
    return np.mean(s2), np.std(s2), acceptance/N
# end function

# Run an emsemble of random walks, keep track of mean value, error and
# the acceptance rate
steps = np.array([5, 10, 15, 20, 25, 30])

means = np.zeros(len(steps))
stds = np.zeros(len(steps))
A = np.zeros(len(steps))

for i in range(0,len(steps)):
    means[i], stds[i], A[i] = self_avoiding_ensemble(steps[i],N=100000)

########################################
#
# 1b) Plot <s^2> vs N with std as error bars
#
########################################

fig, ax = plt.subplots()

ax.errorbar(steps,means,yerr=stds,fmt="o")

# Format plot
ax.set_xlim(steps.min() - 0.1*steps.min(),steps.max() + 0.1*steps.max());
ax.set_xlabel("N")
ax.set_ylabel(r"$<s^2>$")
fig.tight_layout()

fig.savefig("1b.png")

########################################
#
# 1c) Plot log<s^2> vs logN and find the slope of the line
#
########################################

# Make the plot
fig, ax = plt.subplots()

ax.plot(np.log10(steps),np.log10(means),"o-")

# Format plot
#ax.set_xlim(steps.min() - 0.1*steps.min(),steps.max() + 0.1*steps.max());
ax.set_xlabel("logN")
ax.set_ylabel(r"log$<s^2>$")
fig.tight_layout()

fig.savefig("1c.png")

# Fit the line with a simple line
fit = np.polyfit(np.log10(steps),np.log10(means),1)

# Print results to the user
print("Linear Fit of log<s^2> vs log10(N):\n")
print("log<s^2> = %.3lf * log10(N) + %.3lf.\n" % (fit[0],fit[1]))

########################################
#
# 1d) Plot A/10000 vs N
#
########################################

fig, ax = plt.subplots()

ax.plot(steps,A/10000.,"o-")

# Format plot
ax.set_xlim(steps.min() - 0.1*steps.min(),steps.max() + 0.1*steps.max());
ax.set_xlabel("N")
ax.set_ylabel(r"A/10000")
fig.tight_layout()

fig.savefig("1d.png")