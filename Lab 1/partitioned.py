from time import perf_counter
import numpy as np
from numpy.random import Generator, PCG64, MT19937
import matplotlib.pyplot as plt


def chisquare_test(array, bins):
    f_observed = np.histogram(array, bins)  # Split input array into M equal bins
    f_expected = np.ones(bins) * (len(array) / bins)  # Assume equal spacing in bins
    chi_square_array = [0] * bins  # Placeholding empty array
    for i in range(len(array)):
        chi_square_array[i] = (f_observed[i] - f_expected[i]) ** 2 / f_expected[i]
        # chi square equation

    return np.sum(chi_square_array)

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
    Msumarray = np.empty(time)
    Rsumarray = np.empty(time)
    Meanarray = np.empty((time,2))

    for i in range(time):
        r = rng.integers(1,N,1)[0]
        Marray[r], Rarray[r] = Rarray[r], Marray[r]
        Msumarray[i], Rsumarray[i] = Marray.sum(), Rarray.sum()
        Meanarray[i,0] = np.mean(Msumarray[:i])
        Meanarray[i,1] = np.mean(Rsumarray[:i])
        
    plt.plot(tarray, Msumarray, label = 'LHS')
    plt.plot(tarray, Rsumarray, label = 'RHS')
    plt.plot(tarray, Meanarray[:,0],label = 'Mean LHS', linestyle = '--')
    plt.plot(tarray, Meanarray[:,1], label = 'Mean RHS', linestyle = '--')
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime) * 1e3, 4)

    plt.xlabel("Time (arb.)")
    plt.ylabel("Number of particles")
    plt.legend()
    plt.tight_layout()
    plt.hlines(1/2*N,0,time,'lightgrey','--')
    plt.savefig(fname = "Lab 1/IMAGES/PCG64_Lec_Ex")
    print("Simulation time = {}ms".format(elapsed_time))

def simple_partition_PROB(N, M, time, prob, plot_count):

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
    Meanarray = np.empty((time,2))
    
    for i in range(time):
        r = rng.random() # Create random float between 0 and 1
        index = rng.choice(M,1) # Create random index choice
        if 0 < r < prob: # If random float is within probability of moving
            Marray[index], Rarray[index] = 0 , 1 # Move a particle from LHS to RHS
        else: 
            Marray[index], Rarray[index] = 1 , 0 # Move a particle from RHS to LHS
        Msumarray[i], Rsumarray[i] = Marray.sum(), Rarray.sum()
        Meanarray[i,0] = np.mean(Msumarray[:i])
        Meanarray[i,1] = np.mean(Rsumarray[:i])
        # Sum particle no.s at each step for plotting
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime) * 1e3, 4) # Time taken for loops

    axs[plot_count].plot(tarray, Msumarray, label = 'LHS')
    axs[plot_count].plot(tarray, Rsumarray, label = 'RHS')
    axs[plot_count].plot(tarray, Meanarray[:,0],label = 'Mean LHS', linestyle = '--')
    axs[plot_count].plot(tarray, Meanarray[:,1], label = 'Mean RHS', linestyle = '--')
    axs[plot_count].hlines(int(prob*N),0,time,'lightgrey','--')
    axs[plot_count].hlines(int((1-prob)*N),0,time,'lightgrey','--')
    axs[plot_count].set_title('$P(LHS \Rightarrow RHS) = {}$'.format(prob))
    plot_count+=1
        
    fig.supxlabel("Time (arb.)")
    fig.supylabel("Number of particles")
    axs[0].legend(bbox_to_anchor=(0,1.1,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
    fig.tight_layout()
    #print("Simulation time = {}ms".format(elapsed_time))

rng = Generator(PCG64(seed=2010)) # Change accordingly
#simple_partition(200,200,4000) 

prob = 0.25
plot_count = 0
fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize = (10,10))
while prob < 1:
    simple_partition_PROB(200,200,4000,prob, plot_count)
    plot_count +=1
    prob+=0.25
plt.savefig("Lab 1/IMAGES/PCG64_Partitioned_Box")