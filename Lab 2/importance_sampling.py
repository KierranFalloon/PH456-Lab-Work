import math
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

    
def importance_sampling_integral(func, weighing_func, lim_array, sample_points):
    
    """ Integrates function using importance sampling, using the metropolis method to take 
    nonuniform samples of the weighing function.

    Args:
        func (function): Defined function that returns function to be integrated 
        weighing_func (function): Defined normalised sampling weighing function
        nonuniform_func (func): Defined function 
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        sample_points (int): Number of points to sample

    Returns:
        integral (float): The estimated integral
    """
    start_time = perf_counter() # Timing the function
    dimensions = int(len(lim_array) / 2) # Determines the number of integrals
    func_vals = 0
    rand_vals = np.empty((sample_points, dimensions))
    norm_rand_vals = np.empty_like(rand_vals)
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
    
    ### Finding ideal Delta value
    delta_array = np.arange(1,sample_points,1)
    accepted_array = np.empty_like(delta_array)
    #rand_vals = np.arange(0, sample_points)

    for d in range(len(delta_array)): # Testing optimal delta in range
        placeholder, accepted_array[d], placeholder_2 = metropolis_algorithm(weighing_func, rand_vals, delta_array[d]) # perform metropolis algorithm
    idx = (np.abs(accepted_array - (0.5*sample_points))).argmin() # Find delta closest to 50% acceptance
    
    #plt.scatter(delta_array, accepted_array, marker = 'o', color = 'green', label = 'Accepted')
    #plt.scatter(delta_array[idx], accepted_array[idx], marker = 'x', color = 'red', label = 'Ideal $\delta= {}$'.format(delta_array[idx]))
    #plt.xlabel('Delta value')
    #plt.ylabel('No. of values')
    #plt.legend()
    #plt.show()
    ###
    
    norm_rand_vals, a, b = metropolis_algorithm(weighing_func, rand_vals, delta_array[idx]) # Use ideal delta
    function_plots(func, weighing_func, norm_rand_vals, lim_array)
    
    #plt.hist(norm_rand_vals, 100)
    #plt.hist(rand_vals)
    #plt.hist(norm_rand_vals, 10)
    #plt.xlim(norm_rand_vals.min(),norm_rand_vals.max())
    #plt.show()

    for i in range(sample_points): # Loop through number of points again
        func_vals += (func(*norm_rand_vals[i]) / weighing_func(*norm_rand_vals[i]))
        # Determine sum for function value by unpacking each random [xi,yi,zi]
        # Determine sum for variance values similarly
    
    integral = 1/sample_points * func_vals

    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time
    
    return integral, time # Return values

def metropolis_algorithm(sampling_func, rand_vals_array, delta):
    
    """ Function implementing the metropolis algorithm for random walks

    Args:
        sampling_func (function): Defined sampling function
        rand_vals_array (array): Array of pseudo-random floats
        delta (float): Delta as defined in the metropolis algorithm

    Returns:
        (array): Array of non-uniform values from random walk with sampling function
    """
    accepted = 0
    counter = 0
    new_rand_vals = rand_vals_array.copy() # So that each test it is not changed
    index = np.random.randint(0, len(rand_vals_array), 1) # arb. stating point
    
    while counter < len(rand_vals_array): # full circle around array
        
        trial_val = rand_vals_array[index-1] + np.random.randint(-1*delta, delta, 1)
        w = sampling_func(trial_val) / sampling_func(rand_vals_array[index-1])

        if w >= 1:
            new_rand_vals[index] = trial_val
            accepted += 1
            
        else:
            r = np.random.uniform(0,1,1)
            
            if r <= w:
                new_rand_vals[index] = trial_val
                accepted += 1
                
            else:
                new_rand_vals[index] = rand_vals_array[index-1]
        counter += 1
        
    return new_rand_vals, accepted, delta

def function_plots(integrand, weighing_func, x, lim_array):

    plt.figure()
    plt.plot(x, integrand(x), marker = 'x', linestyle = 'None', label = 'Integrand function', color = 'orange')
    plt.plot(x, weighing_func(x), marker = 'x', linestyle = 'None', label = 'Weighing function', color = 'blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim(lim_array[0], lim_array[1])
    plt.legend()
    plt.show()

## Lecture example, to test the integrator
def integrand_1(x):
    return np.exp(-(x**2))

def weighing_func_1(x):
    A = (1 / (1-np.exp(-1)))
    return A * np.exp(-x)

integral, time = importance_sampling_integral(integrand_1,
                                              weighing_func_1,
                                              [0,1],
                                              100)
print(integral, time)

## 5a). approx 3.54

def integrand_2(x):
    return 2*np.exp(-(x**2))

def weighing_func_2(x):
    A = (1 / (2*(1-np.exp(-10))))
    return  (A * np.exp(-(np.abs(x))))

integral2, time2 = importance_sampling_integral(integrand_2,
                                                weighing_func_2,
                                                [-10,10],
                                                100)

print(integral2)

## 5b). = 3
def integrand_3(x):
    return 1.5*np.sin(x)

def weighing_func_3(x):
    A = 3 / (2*np.pi)
    return A * (4/(np.pi**2))*x*(np.pi - x)

integral3, time3 = importance_sampling_integral(integrand_3, 
                                                weighing_func_3, 
                                                [0,np.pi], 
                                                100)
print(integral3)