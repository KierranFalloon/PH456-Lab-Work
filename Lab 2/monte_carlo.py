import math
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def MonteCarlo_Integrator(func, lim_array, sample_points):
    
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
    func_vals, var_vals = 0, 0
    rand_vals = np.empty((sample_points, dimensions)) # Empty random vals matrix size (sample_points rows, dimensions columns)
    
    j=0 # Dimensions counter
    k=0 # Another counter for indexing
    area = 1 # area initial value
    
    while j <= dimensions+(dimensions-1): # Loop through limits via auto counting dimensions
        area *= lim_array[j+1] - lim_array[j] 
        # area = Previous difference in limits * this one, for all limits
        for i in range(sample_points): # Loop through number of points
            rand_vals[i, k] = np.random.uniform(lim_array[j], lim_array[j+1], 1)
            # Create random number and store it in the array, such that each random number
            # is within the input limits
        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit, 
        k+=1 # Index just for assigning rand_vals array

    for i in range(sample_points): # Loop through number of points again
        func_vals += func(*rand_vals[i]) # Determine sum for function value by unpacking each random [xi,yi,zi]
        var_vals += (func(*rand_vals[i]))**2 # Determine sum for variance values similarly
    integral = area/sample_points * func_vals
    # Collate terms in integration estimation
    variance = ( 1/sample_points * ((var_vals) - (func_vals**2)) )
    root_mean_squared = np.sqrt(variance)
    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time
    
    return integral, variance, root_mean_squared, time # Return values

def function1(x):
    return 2

def function2(x):
    return -x

def function3(x): 
    return x**2

def function4(x,y):
    return x*y + x

integral1, variance1, rms1, time1 = MonteCarlo_Integrator(function1,[0,1],10000)
print(integral1, variance1, rms1, time1)

integral2, variance2, rms2, time2 = MonteCarlo_Integrator(function2,[0,1],10000)
print(integral2, variance2, rms2, time2)

integral3, variance3, rms3, time3 = MonteCarlo_Integrator(function3,[-2,2],10000)
print(integral3, variance3, rms3, time3)

integral4, variance4, rms4, time4 = MonteCarlo_Integrator(function4, [0,1,0,1], 10000)
print(integral4, variance4, rms4, time4)

# Use your subroutine to evaluate the size of the region enclosed
# within an n-sphere of radius 2.0, for n = 3 (i.e. the volume of a ball of
# radius 1.5) and n = 5.

def sphere_func_3(x,y,z):
    
    if np.sqrt(x**2 + y**2 + z**2) <= 2:
        return 1
    else:
        return 0

def sphere_func_5(x,y,z,rho,epsilon):
    
    if np.sqrt(x**2 + y**2 + z**2 + rho**2 + epsilon**2) <= 2:
        return 1
    else:
        return 0

integral5, variance5, rms5, time5 = MonteCarlo_Integrator(sphere_func_3, [-2,2,-2,2,-2,2], 10000)
integral6, variance6, rms6, time6 = MonteCarlo_Integrator(sphere_func_5, [-2,2,-2,2,-2,2,-2,2,-2,2], 10000)
print(integral5, variance5, rms5, time5)
print(integral6, variance6, rms6, time6)

