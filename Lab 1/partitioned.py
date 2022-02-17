from time import perf_counter
import numpy as np
from numpy.random import Generator, PCG64, MT19937
import matplotlib.pyplot as plt


def chisquare_test(array, bins):
    """chi-squared test

    Args:
        array: Array to be tested
        bins [int]: Number of bins 'array' is to be split into

    Returns:
        [float]: chi_1-squared value
    """
    f_observed = np.histogram(array, bins)  # Split input array into M equal bins
    f_expected = np.ones(bins) * (len(array) / bins)  # Assume equal spacing in bins
    chi_square_array = [0] * bins  # Placeholding empty array
    for i in range(len(array)):
        chi_square_array[i] = (f_observed[i] - f_expected[i]) ** 2 / f_expected[i]
        # chi square equation

    return np.sum(chi_square_array)

def simple_partition(generator, seed, total_particles, lhs_particles, time):
    """ Simple partitioned box simulation, where particles movement is dictated only by
    the probability of moving to the other side of the partition. This method randomly
    swaps particles to the other side depending on a pseudo random number generator

    Args:
        generator (PCG64, MT19937): Type of generator for pseudo-random numbers
        seed (int): seed for generator
        total_particles (int): Total number of particles
        lhs_particles (int): Number of particles on the LHS initially, <= total_particles
        time (int): Simulation time (arb. units)

    Returns:
        Graph of simulated partitioned box
    """
    assert generator in [PCG64,MT19937], 'rng2 - Valid bitgenerator must be used'

    # Get names as strings from np.random class
    bit_generator = generator.__name__
    rng = Generator(generator(seed = seed))
    plt.figure()

    funcstarttime = perf_counter()
    m_array = np.ones(lhs_particles)
    r_array = np.zeros_like(m_array)
    t_array = np.arange(0,time,1)
    m_sumarray = np.empty(time)
    r_sumarray = np.empty(time)
    mean_array = np.empty((time,2))

    for i in range(time):
        r = rng.integers(1,total_particles,1)[0]
        m_array[r], r_array[r] = r_array[r], m_array[r]
        m_sumarray[i], r_sumarray[i] = m_array.sum(), r_array.sum()
        mean_array[i,0] = np.mean(m_sumarray[:i])
        mean_array[i,1] = np.mean(r_sumarray[:i])

    plt.plot(t_array, m_sumarray, label = 'LHS')
    plt.plot(t_array, r_sumarray, label = 'RHS')
    plt.plot(t_array, mean_array[:,0],label = 'Mean LHS', linestyle = '--')
    plt.plot(t_array, mean_array[:,1], label = 'Mean RHS', linestyle = '--')
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime) * 1e3, 4)

    plt.xlabel("Time (arb.)")
    plt.ylabel("Number of particles")
    plt.legend()
    plt.hlines(1/2*total_particles,0,time,'lightgrey','--')
    plt.tight_layout()
    #plt.savefig(fname = "Lab 1/IMAGES/{}_{}_Lec_Ex".format(bit_generator, seed))

    return print("Simulation time = {}ms".format(elapsed_time))

def simple_partition_prob(generator, seed, total_particles, lhs_particles, time, prob, plot_count):
    """Simple partitioned box simulation, where particles movement is dictated only by
    the user input probability of moving to the other side of the partition. This method randomly
    swaps particles to the other side depending on a pseudo random number generator

    Args:
        generator (PCG64, MT19937): Type of generator for pseudo-random numbers
        seed (int): seed for generator
        total_particles (int): Total number of particles
        lhs_particles (int): Number of particles on the LHS initially, <= total_particles
        time (int): Simulation time (arb. units)
        prob (float): Probability of a particle moving from the LHS to the RHS, <=1
        plot_count (int): Number of plots, dependent on loop

    Returns:
        Graph of simulated partitioned box
    """
    assert generator in [PCG64,MT19937], 'rng2 - Valid bitgenerator must be used'

    rng = Generator(generator(seed = seed))

    funcstarttime = perf_counter()
    m_array = np.ones(lhs_particles) # Assigning '1' to be the LHS state
    r_array = np.zeros_like(m_array) # Assigning '0' to be the RHS state
    t_array = np.arange(0,time,1) # Arbitrary time array for plotting
    m_sumarray = np.zeros(time) # Sum array for plotting
    r_sumarray = np.zeros(time)
    mean_array = np.empty((time,2))

    for i in range(time):
        r = rng.random() # Create random float between 0 and 1
        index = rng.choice(lhs_particles,1) # Create random index choice
        if 0 < r < prob: # If random float is within probability of moving
            m_array[index], r_array[index] = 0 , 1 # Move a particle from LHS to RHS
        else:
            m_array[index], r_array[index] = 1 , 0 # Move a particle from RHS to LHS
        m_sumarray[i], r_sumarray[i] = m_array.sum(), r_array.sum()
        mean_array[i,0] = np.mean(m_sumarray[:i])
        mean_array[i,1] = np.mean(r_sumarray[:i])
        # Sum particle no.s at each step for plotting
    funcendtime = perf_counter()
    elapsed_time = np.round((funcendtime - funcstarttime) * 1e3, 4) # Time taken for loops

    axs[plot_count].plot(t_array, m_sumarray, label = 'LHS')
    axs[plot_count].plot(t_array, r_sumarray, label = 'RHS')
    axs[plot_count].plot(t_array, mean_array[:,0],label = 'Mean LHS', linestyle = '--')
    axs[plot_count].plot(t_array, mean_array[:,1], label = 'Mean RHS', linestyle = '--')
    axs[plot_count].hlines(int(prob*total_particles),0,time,'lightgrey','--')
    axs[plot_count].hlines(int((1-prob)*total_particles),0,time,'lightgrey','--')
    axs[plot_count].set_title('$P(LHS \Rightarrow RHS) = {}$'.format(prob))
    plot_count+=1

    fig.supxlabel("Time (arb.)")
    fig.supylabel("Number of particles")
    axs[0].legend(bbox_to_anchor=(0,1.1,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
    fig.tight_layout()

    print("Simulation time = {}ms".format(elapsed_time))

# Generate data

#simple_partition(PCG64, 2010, 200, 200, 4000)
#simple_partition(MT19937, 2010, 200, 200, 4000)
#simple_partition(PCG64, 15832, 200, 200, 4000)
#simple_partition(PCG64, 15832, 200, 200, 4000)
#simple_partition(PCG64, 6, 200, 200, 4000)
#simple_partition(MT19937, 15832, 200, 200, 4000)
#simple_partition(MT19937, 6, 200, 200, 4000)

Prob = 0.25
Plot_count = 0
fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize = (10,10))
while Prob < 1:
    #simple_partition_prob(PCG64, 6, 200, 200, 4000, Prob, Plot_count)
    #simple_partition_prob(MT19937, 6, 200, 200, 4000, Prob, Plot_count)
    Plot_count +=1
    Prob+=0.25
