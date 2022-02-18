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

    f_observed = np.histogram(array, bins)[0]  # Split input array into M equal bins
    f_expected = np.ones(bins) * (len(array) / bins)  # Assume equal spacing in bins
    chi_1_square_array = [0] * bins  # Placeholding empty array
    for i in range(len(array)):
        chi_1_square_array[i] = (f_observed[i] - f_expected[i]) ** 2 / f_expected[i]
        # chi_1 square equation

    return np.sum(chi_1_square_array)

def rng_tests(generator_1, generator_2, seed_1, seed_2, number_of_points):

    """ Produces graphs to test the effect of seed on generators,
    uniformity (chi-squared value) and sequential correlation
    (shifted value plots) of random number arrays of length number_of_points,
    generated using generator_1 with seed_1 and generator_2 with seed_2.

    Args:
        generator_1 (PCG64, MT19937): Type of generator for first set of numbers
        generator_2 (PCG64, MT19937): Type of generator for first set of numbers
        seed_1 (int): seed for first set of numbers
        seed_2 (int): seed for second set of numbers
        number_of_points (int): length of arrays to be generated

    Returns:
        Graphs, statistics, times:

                        Produces 3 graphs:
                        Generator 1 against Generator 2
                        Generator 1 against Generator 1 with shifted values
                        Generator 2 against Generator 2 with shifted values

                        Each with chi_1 squared, cross corrrelation and pearson
                        product moment correlation coefficient values

                        Prints time taken for each generator
    """

    assert generator_1 in [PCG64,MT19937], 'rng1 - Valid bitgenerator must be used'
    assert generator_2 in [PCG64,MT19937], 'rng2 - Valid bitgenerator must be used'

    # Get names as strings from np.random class
    bit_generator_1 = generator_1.__name__
    bit_generator_2 = generator_2.__name__

    # Define random number generators using input
    rng1 = Generator(generator_1(seed = seed_1))
    rng2 = Generator(generator_2(seed = seed_2))

    # Pseudo-random, uniform range between 0-1 with defined seed
    x_1starttime = perf_counter()
    x_1 = rng1.random(number_of_points)
    x_1endtime = perf_counter()

    time_1 = x_1endtime - x_1starttime # Time taken to generate

    x_15 = np.roll(x_1, 10)  # Move each value of x_1 10 spaces along

    # Pseudo-random, uniform range between 0-1 with defined seed
    x_2starttime = perf_counter()
    x_2 = rng2.random(number_of_points)
    x_2endtime = perf_counter()

    time_2 = x_2endtime - x_2starttime

    x_25 = np.roll(x_2, 10)

    ####################################
    # Statistical tests
    ####################################

    chi_1 = chisquare_test(x_1, number_of_points) # chi_1square tests from defined function
    chi_2 = chisquare_test(x_2, number_of_points)
    # ( chi_1square does not change for shifting values )
    # Cross correlation between
    correl_initial = np.around(np.correlate(x_1, x_2)[0], 2) # Initial arrays
    correl1 = np.around(np.correlate(x_1, x_15)[0], 2) # Initial array and shifted values
    correl2 = np.around(np.correlate(x_2, x_25)[0], 2)
    # Pearson product moment correlation coefficient between
    pearsoncoeff_initial = np.round(np.corrcoef(x_1, x_2)[0, 1], 5) # Initial arrays
    pearsoncoeff1 = np.round(np.corrcoef(x_1, x_15)[0, 1], 5) # Initial array and shifted values
    pearsoncoeff2 = np.round(np.corrcoef(x_2, x_25)[0, 1], 5)

    ####################################
    # Plotting routines
    ####################################

    # Plotting initial arrays against eachother, showing chi_1-squared values

    plt.figure(figsize=(6, 6)) # Formatted to be suitable to show on report
    plt.scatter(x_1, x_2, marker="x")
    plt.title(
        r"Cross-correlation $\approx %1.2f, \rho_{XY}\approx %1.4f$"
        % (correl_initial, pearsoncoeff_initial),
        fontsize=10,)
    plt.xlabel(r"{} seed = {}, $\chi_1^2 = {}$".format(bit_generator_1, seed_1, chi_1))
    plt.ylabel(r"{} seed = {}, $\chi_1^2 = {}$".format(bit_generator_2, seed_2, chi_2))
    plt.tight_layout()
    #plt.savefig(fname = '{}_{}_{}_{}'.format(bit_generator_1, bit_generator_2, seed_1, seed_2))
    plt.show()

    # Shifted value comparison ( sequential correlation test )

    # Plotting each initial array against the corresponding shifted values array
    plt.figure(figsize=(6, 6))
    plt.scatter(x_1, x_15, marker="x")
    plt.xlabel("$x_n$")
    plt.ylabel("$x_{n+10}$")
    plt.title(
        r"Cross-correlation $\approx %1.2f$, $\chi_1^2 \approx %s, \rho_{XY}\approx %1.4f$"
        % (correl1, chi_1, pearsoncoeff1),
        fontsize=10,
    ) # Showing all statistics
    #plt.suptitle(bit_generator_1)
    plt.tight_layout()
    #plt.savefig(fname = '{}_{}'.format(bit_generator_1,seed_1))
    plt.show()

    # Shifted value comparison ( sequential correlation test )
    plt.figure(figsize=(6, 6))
    plt.scatter(x_2, x_25, marker="x")
    plt.xlabel("$x_n$")
    plt.ylabel("$x_{n+10}$")
    plt.title(
        r"Cross-correlation $\approx %1.2f$, $\chi_1^2 \approx %s, \rho_{XY}\approx %1.4f$"
        % (correl2, chi_2, pearsoncoeff2),
        fontsize=10,
    ) # Showing all statistics
    #plt.suptitle(bit_generator_2)
    plt.tight_layout()
    #plt.savefig(fname = '{}_{}'.format(bit_generator_2,seed_2))
    plt.show()

    return print('\nTimes:\n{} seed {} = {}µs, \n{} seed {} = {}µs\n'
                 .format(bit_generator_1, seed_1, time_1*1e6, bit_generator_2, seed_2, time_2*1e6))

def shift_comparison(generator_1, generator_2, seed_1, seed_2, number_of_points):

    """ Generates graphs to compare the effect of shifted index value on both
    cross correlation and the pearson product moment correlation coefficient
    for the input generators and seeds.

    Args:
        generator_1 (PCG64, MT19937): Type of generator for first set of numbers
        generator_2 (PCG64, MT19937): Type of generator for first set of numbers
        seed_1 (int): seed for first set of numbers
        seed_2 (int): seed for second set of numbers
        number_of_points (int): length of arrays to be generated

        Returns:
        Graphs, statistics, times:

                        Produces 2 graphs:
                        Cross-correlation against shifted index for Generator 1 and Generator 2
                        on a subplot
                        Pearson product moment correlation coefficient against shifted index
                        for Generator 1 and Generator 2 on a subplot
                        Each plot contains the mean of the non-peak section, standard deviation,
                        and the maximum deviation of this section.

                        Prints time taken for each generator
    """

    assert generator_1 in [PCG64,MT19937], 'rng1 - Valid bitgenerator must be used'
    assert generator_2 in [PCG64,MT19937], 'rng2 - Valid bitgenerator must be used'

    # Get names as strings from np.random class
    bit_generator_1 = generator_1.__name__
    bit_generator_2 = generator_2.__name__

    # Define random number generators using input
    rng1 = Generator(generator_1(seed = seed_1))
    rng2 = Generator(generator_2(seed = seed_2))

    #####################################################
    # Comparing shifted value vs cross-correlation values
    #####################################################

    shift = np.arange(0, number_of_points+1, 1) # array for the x-axis
    correlation_1 = np.zeros(number_of_points+1)
    chi_1 = np.zeros_like(correlation_1)
    pearsoncoeff_1 = np.zeros_like(correlation_1)

    # Pseudo-random, uniform range between 0-1 with defined seed
    x_1starttime = perf_counter()
    x_1 = rng1.random(number_of_points)
    x_1endtime = perf_counter()

    time_1 = x_1endtime - x_1starttime # Time taken to generate


    for i in range(number_of_points+1):
        x_15 = np.roll(x_1, shift[i]) # Shift values by 1
        correlation_1[i] = np.correlate(x_1, x_15) # Calculate and append stats
        chi_1[i] = chisquare_test(x_15, number_of_points)
        pearsoncoeff_1[i] = np.around(np.corrcoef(x_1, x_15)[0, 1], 5)

    correlation_2 = np.zeros(number_of_points+1)
    chi_2 = np.zeros_like(correlation_2)
    pearsoncoeff_2 = np.zeros_like(correlation_2)

    # Pseudo-random, uniform range between 0-1 with defined seed
    x_2starttime = perf_counter()
    x_2 = rng2.random(number_of_points)
    x_2endtime = perf_counter()

    time_2 = x_2endtime - x_2starttime

    for i in range(number_of_points+1):
        x_25 = np.roll(x_2, shift[i]) # Shift value by 1
        correlation_2[i] = np.correlate(x_2, x_25) # Calculate and append stats
        chi_2[i] = chisquare_test(x_25, number_of_points)
        pearsoncoeff_2[i] = np.around(np.corrcoef(x_2, x_25)[0, 1], 5)

    # Values for visualisation
    range1 = np.round(correlation_1[1:number_of_points].max() - correlation_1.min(), 2)
    range2 = np.round(correlation_2[1:number_of_points].max() - correlation_2.min(), 2)
    std1 = np.round(np.std(correlation_1[1:number_of_points]), 3)
    std2 = np.round(np.std(correlation_2[1:number_of_points]), 3)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(shift, correlation_1)
    ax1.set_title(bit_generator_1)
    ax1.hlines(correlation_1.max(), shift[0], shift[number_of_points],
               color = 'grey', linestyle = '--',
               label = "Max = {}".format(np.round(correlation_1.max(), 1)))
    ax1.hlines(np.mean(correlation_1[1:number_of_points]),0,number_of_points,
               color = "r", linestyle = "--",
               label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
               .format(np.round(np.mean(correlation_1[1:number_of_points]), 1), std1, range1))

    ax2.plot(shift, correlation_2)
    ax2.set_title(bit_generator_2)
    ax2.hlines(correlation_2.max(), shift[0], shift[number_of_points],
               color = 'grey', linestyle = '--',
               label = "Max = {}".format(np.round(correlation_2.max(), 1)))
    ax2.hlines(np.mean(correlation_2[1:number_of_points]),0,number_of_points,
               color = "r",linestyle = "-.",
               label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
               .format(np.round(np.mean(correlation_2[1:number_of_points]), 1), std2, range2))
    ax2.set_xlabel("Lag $i$")

    fig.supylabel("Cross-correlation between $x_n~&~x_{n+i}$")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    #plt.savefig(fname = '{}{}_{}{}_correl'.format(bit_generator_1,seed_1,bit_generator_2,seed_2))
    plt.show()

    ################################################
    # Comparing shifted value vs pearson values
    ################################################

    pearson_range1 = np.round(pearsoncoeff_1[1:number_of_points].max() - pearsoncoeff_1.min(), 4)
    pearson_range2 = np.round(pearsoncoeff_2[1:number_of_points].max() - pearsoncoeff_2.min(), 4)
    pearson_std1 = np.round(np.std(pearsoncoeff_1[1:number_of_points]), 5)
    pearson_std2 = np.round(np.std(pearsoncoeff_2[1:number_of_points]), 5)

    fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(shift, pearsoncoeff_1)
    ax1.set_title(bit_generator_1)
    ax1.hlines(pearsoncoeff_1.max(), shift[0], shift[number_of_points],
               color = 'grey', linestyle = '--',
               label = "Max = {}".format(np.round(pearsoncoeff_1.max(), 2)))
    ax1.hlines(np.mean(pearsoncoeff_1[1:number_of_points]),0,number_of_points,
               color = "r", linestyle = "--",
               label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
               .format(np.round(np.mean(pearsoncoeff_1[1:number_of_points])*-1,2),
                       pearson_std1, pearson_range1))

    ax2.plot(shift, pearsoncoeff_2)
    ax2.set_title(bit_generator_2)
    ax2.hlines(pearsoncoeff_2.max(), shift[0], shift[number_of_points],
               color = 'grey', linestyle = '--',
               label = "Max = {}".format(np.round(pearsoncoeff_2.max(), 2)))
    ax2.hlines(np.mean(pearsoncoeff_2[1:number_of_points]),0,number_of_points,
               color = "r", linestyle = "--",
               label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
               .format(np.round(np.mean(pearsoncoeff_2[1:number_of_points])*-1, 2),
                       pearson_std2, pearson_range2))

    ax2.set_xlabel("Lag $i$")
    fig2.supylabel(r"$\rho_{XY}$ between $x_n~&~x_{n+i}$")
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    #plt.savefig(fname = '{}{}_{}{}_pearson'.format(bit_generator_1,seed_1,bit_generator_2,seed_2))
    plt.show()

    return print('\nTimes:\n{} seed {} = {}µs, \n{} seed {} = {}µs\n'
                 .format(bit_generator_1, seed_1, time_1*1e6, bit_generator_2, seed_2, time_2*1e6))


# Generate results
#rng_tests(PCG64,PCG64,6,15832,2000)
#rng_tests(PCG64,MT19937,6,6,2000)
#rng_tests(MT19937,MT19937,5255,7381,2000)
#rng_tests(PCG64, MT19937, 6, 7381, 2000)
rng_tests(MT19937, MT19937, 6, 15832, 2000)


#shift_comparison(PCG64, MT19937, 15832, 15832, 1000)
#shift_comparison(PCG64, MT19937, 6, 6, 1000)
#shift_comparison(PCG64, MT19937, 7381, 7381, 1000)
#shift_comparison(PCG64, MT19937, 5255, 5255, 1000)
