import numpy as np
from numpy.random import Generator, PCG64, MT19937
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.stats import chisquare

def chisquare_test(array, bins):
    f_observed, edges = np.histogram(array, bins) # Split input array into M equal bins
    f_expected = np.ones(bins)*( len(array) / bins ) # Assume equal spacing in bins
    chi_square_array = [0]*bins # Placeholding empty array
    for i in range(len(array)):
        chi_square_array[i] = ( (f_observed[i] - f_expected[i])**2 / f_expected[i]) # chi square equation
    
    return np.sum(chi_square_array)

    """Write a program to simulate a partitioned box containing N particles,
initially all on one side of the partition, with an equal probability of
any one particle moving from one side of the partition to the other in
unit time. Present your results graphically as well as textually.
    """

#Generate a random number r from a uniformly distributed set of random numbers in the interval 0 ≤ r < 1.
#Compare r to the current value of the fraction of particles n/N on the left side of the box.
#If r < n/N, move a particle from left to right, that is, let n → n - 1; otherwise, move a particle from right to left.
#Increase the time by 1.

rng = Generator(PCG64(seed=3))

N = 100 # Initial number of particles in the entire box
M = 100 # initial number of particles on the L.H.S
plt.figure()
time = 1000
Marray = np.zeros(time)
Rarray = np.zeros_like(Marray)
tarray = np.zeros_like(Marray)
for t in range(time): # simulating 1000s of time
    r = rng.uniform(0, 1) # Random number between 0 and 1
    frac = M/N # As no other factors are considered, this can be thought of as the probability of moving 
    if r < frac: # if random number is less than the fraction, particle moves from LHS to RHS
        M = M-1
    elif r > frac:
        M = M+1 # otherwise, particle move from RHS to LHS
    else: # if equilibrium, end as no more movement is possible
        break
    Marray[t] = M
    Rarray[t] = 1-M
    tarray[t] = t

plt.plot(tarray,Marray,label = 'LHS')
plt.xlabel("Time (arb.)")
plt.ylabel("Number of particles")
plt.hlines(50,0,time,color = 'grey', linestyle = '--')
plt.legend()
plt.savefig("IMAGES/Simple partitioned box")
plt.show()

