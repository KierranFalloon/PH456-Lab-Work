from logging import root
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
    
def importance_sampling_integral(func, weighing_func, lim_array, sample_points, xi):
    
    """ Integrates function using importance sampling, using the metropolis method to take 
    nonuniform samples of the weighing function.

    Args:
        func (function): Defined function that returns function to be integrated 
        weighing_func (function): Defined normalised sampling weighing function
        nonuniform_func (func): Defined function 
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        sample_points (int): Number of points to sample
        xi (float): Arbitrary initial point on the random walk

    Returns:
        integral (float): The estimated integral
    """
    
    start_time = perf_counter() # Timing the function
    dimensions = int(len(lim_array) / 2) # Determines the number of integrals
    func_vals, var_vals = 0, 0
    # Empty random vals matrix size (sample_points rows, dimensions columns)

    j=0 # Dimensions counter
    k=0 # Another counter for indexing
    ### Finding ideal Delta value
    #initial_position_array = np.arange(lim_array[0],lim_array[1], np.diff(lim_array)/sample_points)
    #accepted_array = np.empty_like(initial_position_array)
    
    #for d in range(len(initial_position_array)): # Testing optimal delta in range
    #    placeholder, accepted_array[d] = metropolis_algorithm(weighing_func, initial_position_array[d], sample_points, dimensions, k) # perform metropolis algorithm
    #idx = (np.abs(accepted_array - (0.5*sample_points))).argmin() # Find delta closest to 50% acceptance
    
    nonuniform_rand_vals = np.empty((sample_points, dimensions))
    while j <= dimensions+(dimensions-1): # Loop through limits via auto counting dimensions
        
        for i in range(sample_points): # Loop through number of points
            nonuniform_rand_vals, acceptance = metropolis_algorithm(weighing_func, xi, sample_points, dimensions, k, lim_array)

        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit
        k+=1 # Index just for assigning rand_vals array
        
    #print('Ratio of values accepted = {}%'.format(int((acceptance/sample_points)*100)))
    
    #plt.scatter(initial_position_array, accepted_array, marker = 'o', color = 'green', label = 'Accepted')
    #plt.scatter(initial_position_array[idx], accepted_array[idx], marker = 'x', color = 'red', label = 'Ideal $x_i= {}$'.format(initial_position_array[idx]))
    #plt.xlabel('Arb. value x')
    #plt.ylabel('No. of values')
    #plt.legend()
    #plt.show()

    #function_plots(func, weighing_func, nonuniform_rand_vals, lim_array, sample_points)
    for i in range(sample_points): # Loop through number of points again
        func_vals += (func(*nonuniform_rand_vals[i]) / weighing_func(*nonuniform_rand_vals[i]))
        var_vals += np.square((func(*nonuniform_rand_vals[i]) / weighing_func(*nonuniform_rand_vals[i])))
        # Determine sum for function value by unpacking each random [xi,yi,zi]
        # Determine sum for variance values similarly
    
    integral = 1/sample_points * func_vals
    variance = 1/sample_points * (1/sample_points * var_vals - 
                                  np.square(1/sample_points * func_vals))
    root_mean_squared = np.sqrt(variance)

    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time

    return integral, time, variance, root_mean_squared # Return values

def metropolis_algorithm(sampling_func, xi, sample_points, dimensions, k, lim_array):
    
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
    nonuniform_rand_vals = np.empty((sample_points, dimensions))
    
    while counter < len(nonuniform_rand_vals): # full circle around array
        
        #xi = np.random.randint(lim_array[0], lim_array[1],1)
        
        trial_val = xi + np.random.normal(xi, 1, 1) # create trial value
        
        if not lim_array[0] <= trial_val <= lim_array[1]:
            continue # condition that value must be within integration limits
        
        w = sampling_func(trial_val) / sampling_func(xi) # calculate ratio

        if w >= 1 or (w<1 and np.random.uniform(0,1,1) <= w): # acceptance conditions
            nonuniform_rand_vals[counter,k] = trial_val
            accepted += 1
                
        else:
            nonuniform_rand_vals[counter,k] = xi # reject trial val and set value 
                                                 # as initial condition

        counter += 1

    return nonuniform_rand_vals, accepted

def function_plots(integrand, weighing_func, x, lim_array, sample_points):
    plt.figure()
    x_array = np.linspace(lim_array[0], lim_array[1], sample_points)
    plt.plot(x_array, integrand(x), label = 'Integrand function', color = 'orange')
    plt.plot(x_array, weighing_func(x), label = 'Weighing function', color = 'blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim(lim_array[0], lim_array[1])
    plt.legend()
    plt.show()

## 5a). approx 3.54

def integrand_1(x):
    return 2*np.exp(-(x**2))

def weighing_func_1(x):
    A = (1 / (2*(1-np.exp(-10))))
    return  (A * np.exp(-(np.abs(x))))

integral1, time1, var1, rms1 = importance_sampling_integral(integrand_1,
                                                weighing_func_1,
                                                [-10,10],
                                                100,
                                                1) # Approx 50% values accepted

print(integral1, time1, var1, rms1)

## 5b). = 3
def integrand_2(x):
    return 1.5*np.sin(x)

def weighing_func_2(x):
    A = 3 / (2*np.pi)
    return A * (4/(np.pi**2))*x*(np.pi - x)

integral2, time2, var2, rms2 = importance_sampling_integral(integrand_2,
                                                weighing_func_2,
                                                [0,np.pi],
                                                100,
                                                1.35) # Approx 70%
print(integral2, time2, var2, rms2)

## Convergence correlation with sampling
def convergence_test(integrand, weighing_func, lim_array, definite_integral, samples, xi):
    
    if definite_integral == None:
        definite_integral = importance_sampling_integral(integrand,
                                                         weighing_func,
                                                         lim_array,
                                                         samples,
                                                         xi)
        # If no definite integral is known it is estimated with high sample size initially
        
    #print("\nRunning convergence test...\n")
    integral = importance_sampling_integral(integrand,
                                            weighing_func,
                                            lim_array,
                                            samples,
                                            xi)
    
    #print('Initial value = {}'.format(integral[0]))
    while not (99 <= (integral[0]/definite_integral)*100 <= 101):
        #print((integral[0]/definite_integral)*100)
        integral = importance_sampling_integral(integrand,
                                                weighing_func,
                                                lim_array,
                                                samples,
                                                xi)
        samples += 10
    samples01 = samples
        
    #print("\n ±1'%' accuracy within {} samples...\nvalue = {}\n".format(samples, integral[0]))

    while not (99.5 <= (integral[0]/definite_integral)*100 <= 100.5):
        #print((integral[0]/definite_integral)*100)
        integral = importance_sampling_integral(integrand,
                                                weighing_func,
                                                lim_array,
                                                samples,
                                                xi)
        samples += 1
    #print("\n ±0.1'%' accuracy within {} samples...\nvalue = {}\n".format(samples, integral[0]))
    return samples01, samples, integral[0]

"""
mc_samples_array = np.empty([2,10])
for b in range(10):
    values = convergence_test(integrand_1,
                              weighing_func_1,
                              [-10,10],
                              3.544907701811032,
                              10,
                              1)
    mc_samples_array[0,b] = values[0]
    mc_samples_array[1,b] = values[1]
print("\nMean ±1 samples = {}".format(np.mean(mc_samples_array[0,:])))
print("Mean ±0.5 samples = {}".format(np.mean(mc_samples_array[1,:])))

mc_samples_array = np.empty([2,10])
for b in range(10):
    values = convergence_test(integrand_2,
                              weighing_func_2,
                              [0,np.pi],
                              3.0,
                              10,
                              1.25)
    mc_samples_array[0,b] = values[0]
    mc_samples_array[1,b] = values[1]
print("\nMean ±1 samples = {}".format(np.mean(mc_samples_array[0,:])))
print("Mean ±0.5 samples = {}".format(np.mean(mc_samples_array[1,:])))
"""