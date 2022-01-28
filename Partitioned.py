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
    