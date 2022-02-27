import math
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

    
def importance_sampling_integral(func, normalised_sampling_func, nonuniform_func, lim_array, sample_points):

    """ Integrates input function via Monte Carlo methods

    Args:
        func (function): Defined function that returns function to be integrated 
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        sample_points (int): Number of points to sample (higher = higher accuracy)

    Returns:
        integral, variance (float): The estimated integral and variance in the estimate
    """
    start_time = perf_counter() # Timing the function
    dimensions = int(len(lim_array) / 2) # Determines the number of integrals
    func_vals = 0
    rand_vals = np.empty((sample_points, dimensions))
    # Empty random vals matrix size (sample_points rows, dimensions columns)

    j=0 # Dimensions counter
    k=0 # Another counter for indexing

    while j <= dimensions+(dimensions-1): # Loop through limits via auto counting dimensions
        # area = Previous difference in limits * this one, for all limits
        for i in range(sample_points): # Loop through number of points
            rand_vals[i, k] = np.random.uniform(lim_array[j], lim_array[j+1], 1)
            # Create random number and store it in the array, such that each random number
            # is within the input limits
        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit
        k+=1 # Index just for assigning rand_vals array
    nonuniform_rand_vals = nonuniform_func(rand_vals)

    for i in range(sample_points): # Loop through number of points again
        func_vals += func(*nonuniform_rand_vals[i]) / normalised_sampling_func(*nonuniform_rand_vals[i])
        # Determine sum for function value by unpacking each random [xi,yi,zi]
        # Determine sum for variance values similarly
        
    integral = 1/sample_points * func_vals 

    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time
    
    return integral, time # Return values


def integrand_1(x):
    return np.exp(-x**2)

def norm_sample_func_1(x):
    A = (1 / (1-np.exp(-1)))
    return A * np.exp(-x)

def nonuniform_func_1(uniform_array):
    A = (1 / (1-np.exp(-1)))
    return -np.log(1 - (uniform_array/A))

integral, time = importance_sampling_integral(integrand_1,norm_sample_func_1,nonuniform_func_1,[0,1],1000)
print(integral)