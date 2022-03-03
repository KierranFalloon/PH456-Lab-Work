import math
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

    
def importance_sampling_integral(func, weighing_func, nonuniform_func, lim_array, sample_points):
    
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
    delta = np.arange(0,lim_array[1]+10,1)
    accepted_array = np.empty_like(delta)

    for d in range(len(delta)): # Testing optimal delta in range
        placeholder, accepted_array[d], placeholder_2 = metropolis_algorithm(weighing_func, rand_vals, delta[d]) # perform metropolis algorithm
    idx = (np.abs(accepted_array - (0.5*sample_points))).argmin()
    
    plt.scatter(delta, accepted_array, marker = 'o', color = 'green', label = 'Accepted')
    plt.scatter(delta[idx], accepted_array[idx], marker = 'x', color = 'red', label = 'Ideal $\delta= {}$'.format(delta[idx]))
    plt.xlabel('Delta value')
    plt.ylabel('No. of values')
    plt.legend()
    plt.show()
    ###
    
    norm_rand_vals, a, b = metropolis_algorithm(weighing_func, rand_vals, delta[idx]) # Use ideal delta
    plt.plot(rand_vals)
    plt.plot(norm_rand_vals)
    plt.show()
    
    scaled_rand_vals = nonuniform_func(norm_rand_vals)

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
        (array): Array of pseuo-random values from random walk with sampling function
    """
    new_rand_vals = np.empty_like(rand_vals_array)
    accepted = 0
    for count in range(len(rand_vals_array)):
        new_rand = rand_vals_array[count] + np.random.uniform(-delta, delta, 1)
        w = sampling_func(new_rand) / sampling_func(rand_vals_array[count])
        if w >= 1:
            new_rand_vals[count] = new_rand
            accepted += 1
        else:
            r = np.random.uniform(0,1,1)
            if r <= w:
                new_rand_vals[count] = new_rand
                accepted += 1
            else:
                new_rand_vals[count] = rand_vals_array[count]
    return new_rand_vals, accepted, delta

## Lecture example, to test the integrator
def integrand_1(x):
    return np.exp(-x**2)

def weighing_func_1(x):
    A = (1 / (1-np.exp(-1)))
    return A * np.exp(-x)

def nonuniform_func_1(uniform_array):
    A = (1 / (1-np.exp(-1)))
    return -np.log(1 - (uniform_array/A))

integral, time = importance_sampling_integral(integrand_1,
                                              weighing_func_1,
                                              nonuniform_func_1,
                                              [0,1],
                                              100)
print(integral)


## 5a). approx 3.54

def integrand_2(x):
    return 2*np.exp(-x**2)

def weighing_func_2(x):
    A = 1 / (2*(1-np.exp(-10)))
    return A * np.exp(-np.abs(x))

def nonuniform_func_2(uniform_array):
    A = 1 / (2*(1-np.exp(-10)))
    return -np.log((uniform_array/A))+10

integral2, time2 = importance_sampling_integral(integrand_2,
                                                weighing_func_2,
                                                nonuniform_func_2,
                                                [-10,10],
                                                100)

print(integral2)

def integrand_3(x):
    return 1.5*np.sin(x)

def weighing_func_3(x):
    A = 3 / (2*np.pi)
    return A * (4/(np.pi**2))*x*(np.pi - x)

def nonuniform_func_2(uniform_array):
    A = 1 / (2*(1-np.exp(-10)))
    return -np.log((uniform_array/A))+10

integral3, time3 = importance_sampling_integral(integrand_3, weighing_func_3, nonuniform_func_2, [0,np.pi], 100)
print(integral3)