from time import perf_counter
import numpy as np
from numpy.random import Generator, PCG64, PCG64
import matplotlib.pyplot as plt


def chisquare_test(array, bins):
    f_observed = np.histogram(array, bins)  # Split input array into M equal bins
    f_expected = np.ones(bins) * (len(array) / bins)  # Assume equal spacing in bins
    chi_square_array = [0] * bins  # Placeholding empty array
    for i in range(len(array)):
        chi_square_array[i] = (f_observed[i] - f_expected[i]) ** 2 / f_expected[i]
        # chi square equation

    return np.sum(chi_square_array)

    """Write a program to simulate a partitioned box containing N particles,
initially all on one side of the partition, with an equal probability of
any one particle moving from one side of the partition to the other in
unit time. Present your results graphically as well as textually.
    """

rng = Generator(PCG64())

def simple_partition(N, M, time):

    plt.figure()

    """Simple partitioned box simulation, where particles movement is dictated only by
    the probability of moving to the other side of the partition. This method uses arrays that
    track particle positions.

    Args:
        N: Total number of particles in the box
        M: Number of particles initially on the L.H.S
        time: Length of time (arb. units)
        frac (0 <= frac <= 1): Probability of a particle moving from the L.H.S to the R.H.S
    """

    funcstarttime = perf_counter()
    Marray = np.ones(M)
    Rarray = np.zeros_like(Marray)
    tarray = np.arange(0,time,1)
    Msumarray = np.zeros(time)
    Rsumarray = np.zeros(time)

    for i in range(time):
        r = rng.integers(1,N,1)[0]
        Marray[r], Rarray[r] = Rarray[r], Marray[r]
        Msumarray[i], Rsumarray[i] = Marray.sum(), Rarray.sum()

    plt.plot(tarray, Msumarray, label = 'LHS')
    plt.plot(tarray, Rsumarray, label = 'RHS')
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime) * 1e3, 4)

    plt.xlabel("Time (arb.)")
    plt.ylabel("Number of particles")
    plt.suptitle('Lecture example, PCG64')
    plt.legend()
    plt.tight_layout()
    print(r"Simulation time = ${}ms$".format(elapsed_time))

def simple_partition_PROB(N, M, time, prob):

    plt.figure()

    """Simple partitioned box simulation, where particles movement is dictated only by
    the probability of moving to the other side of the partition. This method takes input of
    probability of passing from LHS to RHS.

    Args:
        N: Total number of particles in the box
        M: Number of particles on the L.H.S
        time: Length of time (arb. units)
        prob (0 <= prob <= 1): Desired probability of a particle moving from the L.H.S to the R.H.S
    """

    funcstarttime = perf_counter()
    Marray = np.ones(M) # Assigning '1' to be the LHS state
    Rarray = np.zeros_like(Marray) # Assigning '0' to be the RHS state
    tarray = np.arange(0,time,1) # Arbitrary time array for plotting
    Msumarray = np.zeros(time) # Sum array for plotting
    Rsumarray = np.zeros(time)
    
    for i in range(time):
        r = rng.random() # Create random float between 0 and 1
        index = rng.choice(M,1) # Create random index choice
        if 0 < r < prob: # If random float is within probability of moving
            Marray[index], Rarray[index] = 0 , 1 # Move a particle from LHS to RHS
        else: 
            Marray[index], Rarray[index] = 1 , 0 # Move a particle from RHS to LHS
        Msumarray[i], Rsumarray[i] = Marray.sum(), Rarray.sum() 
        # Sum particle no.s at each step for plotting
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime) * 1e3, 4) # Time taken for loops

    plt.plot(tarray, Msumarray, label = 'LHS')
    plt.plot(tarray, Rsumarray, label = 'RHS')
    plt.xlabel("Time (arb.)")
    plt.ylabel("Number of particles")
    plt.suptitle('PCG64')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(r"Simulation time = ${}ms$".format(elapsed_time))

simple_partition_PROB(100,100,1000,0.75)