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
time = 1000
Marray = np.zeros(time)
Rarray = np.zeros_like(Marray)
tarray = np.zeros_like(Marray)

for t in range(time): # simulating 1000s of time
    r = rng.uniform(0, 1) # Random number between 0 and 1
    frac = M/N # As no other factors are considered, this can be thought of as the probability of moving 
    
    if r < frac: # if random number is less than the fraction, particle moves from LHS to RHS
        M = M-1
        R = N-M
    elif r > frac:
        M = M+1 # otherwise, particle move from RHS to LHS
        R = N-M

    Marray[t] = M
    Rarray[t] = R
    tarray[t] = t

plt.figure()
plt.plot(tarray,Marray,label = 'LHS')
plt.plot(tarray,Rarray,label = 'RHS')
plt.xlabel("Time (arb.)")
plt.ylabel("Number of particles")
plt.title("Simple partitioned box simulation")
plt.hlines(0,tarray[0],tarray[time-1], color = 'lightgrey', linestyle = '--')
plt.hlines(N,tarray[0],tarray[time-1], color = 'lightgrey', linestyle = '--')
plt.legend()
plt.savefig("IMAGES/Simple partitioned box")
 

def simple_partition(N,M,time,prob):
    
    """ Simple partitioned box simulation, where particles movement is dictated only by 
    the probability of moving to the other side of the partition. This method tracks only the no.
    of particles on each side of the partition.
    
    Args:
        N: Total number of particles in the box
        M: Number of particles initially on the L.H.S
        time: Length of time (arb. units)
        frac (0 <= frac <= 1): Probability of a particle moving from the L.H.S to the R.H.S
    """

    funcstarttime = perf_counter()
    Marray = np.zeros(time)
    Rarray = np.zeros_like(Marray)
    tarray = np.zeros_like(Marray)
     
    for t in range(time):
        r = rng.uniform(0, 1)
        R = N-M

        if r < prob: # if random number is less than the fraction, particle moves from LHS to RHS
            M -= 1
            R = N-M
            if M >= N:
                M = N
            if R >= N:
                R = N

            if M <= 0:
                M = 0 
            if R <= 0:
                R = 0

        elif r > prob:
            M += 1 # otherwise, particle move from RHS to LHS
            R = N-M
            if M >= N:
                M = N
            if R >= N:
                R = N

            if M <= 0:
                M = 0 
            if R <= 0:
                R = 0

        Marray[t] = M
        Rarray[t] = R
        tarray[t] = t
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime)*1e3,4)

    plt.figure()
    plt.plot(tarray,Marray,label = 'LHS')
    plt.plot(tarray,Rarray,label = 'RHS')
    plt.hlines(0,tarray[0],tarray[time-1], color = 'lightgrey', linestyle = '--')
    plt.hlines(N,tarray[0],tarray[time-1], color = 'lightgrey', linestyle = '--')
    plt.xlabel("Time (arb.)")
    plt.ylabel("Number of particles")
    plt.title('Simple partitioned box simulation\n P(LHS $\longrightarrow$ RHS) = {}'.format(prob))
    plt.legend()
    plt.savefig("IMAGES/Simple partitioned box_{}".format(int(prob*100)))
    plt.show()
    print("Simulation time = ${}m s$".format(elapsed_time))

for i in range(3):
    simple_partition(100,100,1000,0.25*(i+1))