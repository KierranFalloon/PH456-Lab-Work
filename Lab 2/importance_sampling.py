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
    func_vals = 0
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
            nonuniform_rand_vals, acceptance = metropolis_algorithm(weighing_func, xi, sample_points, dimensions, k) # Use ideal delta
            # Create random number and store it in the array, such that each random number
            # is within the input limits
        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit
        k+=1 # Index just for assigning rand_vals array
    print('Ratio of values accepted = {}%'.format(int((acceptance/sample_points)*100)))
    #plt.scatter(initial_position_array, accepted_array, marker = 'o', color = 'green', label = 'Accepted')
    #plt.scatter(initial_position_array[idx], accepted_array[idx], marker = 'x', color = 'red', label = 'Ideal $x_i= {}$'.format(initial_position_array[idx]))
    #plt.xlabel('Arb. value x')
    #plt.ylabel('No. of values')
    #plt.legend()
    #plt.show()
    
    ###

    function_plots(func, weighing_func, nonuniform_rand_vals, lim_array, sample_points)


    for i in range(sample_points): # Loop through number of points again
        func_vals += (func(*nonuniform_rand_vals[i]) / weighing_func(*nonuniform_rand_vals[i]))
        # Determine sum for function value by unpacking each random [xi,yi,zi]
        # Determine sum for variance values similarly
    
    integral = 1/sample_points * func_vals

    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time
    
    return integral, time # Return values

def metropolis_algorithm(sampling_func, xi, sample_points, dimensions, k):
    
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
        
        trial_val = xi + np.random.normal(xi, 1, 1)
        w = sampling_func(trial_val) / sampling_func(xi)

        if w >= 1 or (w<1 and np.random.uniform(0,1,1) <= w):
            nonuniform_rand_vals[counter,k] = trial_val
            accepted += 1
                
        else:
            nonuniform_rand_vals[counter] = xi

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

def integrand_2(x):
    return 2*np.exp(-(x**2))

def weighing_func_2(x):
    A = (1 / (2*(1-np.exp(-10))))
    return  (A * np.exp(-(np.abs(x))))

integral2, time2 = importance_sampling_integral(integrand_2,
                                                weighing_func_2,
                                                [-10,10],
                                                100,
                                                1) # Approx 50% values accepted

print(integral2, time2)

## 5b). = 3
def integrand_3(x):
    return 1.5*np.sin(x)

def weighing_func_3(x):
    A = 3 / (2*np.pi)
    return A * (4/(np.pi**2))*x*(np.pi - x)

integral3, time3 = importance_sampling_integral(integrand_3, 
                                                weighing_func_3, 
                                                [0,np.pi], 
                                                100,
                                                1.35) # Approx 70%
print(integral3, time3)