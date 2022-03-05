from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
    
def importance_sampling_integral(integrand, weighing_func, lim_array, sample_points, initial_value):
    
    """ Integrates function using importance sampling, using the metropolis method to take 
    nonuniform samples of the weighing function.

    Args:
        integrand (function): Defined function that returns function to be integrated 
        weighing_func (function): Defined normalised sampling weighing function
        nonuniform_func (func): Defined function 
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        sample_points (int): Number of points to sample
        initial_value (float): Arbitrary initial point on the random walk

    Returns:
        integral (float): The estimated integral
    """

    start_time = perf_counter() # Timing the function
    dimensions = int(len(lim_array) / 2) # Determines the number of integrals
    func_vals, var_vals = 0, 0

    global j, k
    j = 0 # Dimensions counter, global so metropolis_algorithm can see it 
    k = 0 # Index just for assigning rand_vals array
    array = np.zeros((sample_points, dimensions))
    
    metropolis_params = [weighing_func, array, lim_array, initial_value, 10, 3]
    # Initial value defined outside of function, discards first 10 values then takes every 3rd value after that
    while j <= dimensions+(dimensions-1): # Loop through limits via auto counting dimensions
        
        for i in range(sample_points): # Loop through number of points
            acceptance = metropolis_algorithm(*metropolis_params)

        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit
        k+=1 # Index just for assigning rand_vals array
        
    #print('Ratio of values accepted = {}%'.format(int((acceptance/sample_points)*100)))

    #function_plots(integrand, weighing_func, array, lim_array, sample_points)
    
    for i in range(sample_points): # Loop through number of points again
        func_vals += (integrand(*array[i]) / weighing_func(*array[i]))
        var_vals += np.square((integrand(*array[i]) / weighing_func(*array[i])))
        # Determine sum for function value by unpacking each random [initial_value,yi,zi]
        # Determine sum for variance values similarly
    
    integral = 1/sample_points * func_vals
    variance = 1/sample_points * (1/sample_points * var_vals - 
                                  np.square(1/sample_points * func_vals))
    root_mean_squared = np.sqrt(variance)

    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time

    return integral, time, variance, root_mean_squared # Return values

def metropolis_algorithm(sampling_func, array, 
                         lim_array, initial_value,
                         discard_init, discard_sampling):
    """Implementation of the metropolis-hastings algorithm

    Args:
        sampling_func (function): Defined sampling function
        initial_value (int, float): Arb. starting point value
        sample_points (int): Number of runs
        dimensions (int): Dimensions of integral as defined
        k (int): Global dimension counter, for multidimensional arrays
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        discard_init (int): Number of initial values to discard (reduce correl to initial_value)
        discard_sampling (int): Value defining each (removal_sampling)th value to discard between values

    Returns:
        array (array): Random walk values sampled from the input function
        accepted (int): % of number of new trial points accepted during the walk
    """
    
    accepted = 0
    counter = 0
    
    array[0] = initial_value
    
    iterations = len(array)*discard_sampling + discard_init
    initial_vals = np.random.uniform(lim_array[j], lim_array[j+1],size = iterations)
    sigmas = np.random.normal(size = iterations)

    while counter < iterations: # full circle around array
        trial_val = initial_vals[counter] + sigmas[counter] # create trial value
        
        if not lim_array[0] <= trial_val <= lim_array[1]:
            counter += 1
            continue # condition that value must be within integration limits
        
        w = sampling_func(trial_val) / sampling_func(initial_value)
        
        if w >= 1: # acceptance conditions
            
            initial_value = trial_val
            accepted += 1
            
        elif (w<1 and np.random.uniform() <= w):
            initial_value = trial_val 
            accepted += 1
                                                
        # Discard first (discard_init) values, and only keep every (discard_sampling)nd value
        # after that
        if (counter > discard_init) and (counter % discard_sampling == 0):
            array[(counter - discard_init) // discard_sampling, k] = initial_value
        
        counter += 1

    return (accepted/discard_sampling)-discard_init

def function_plots(integrand, weighing_func, x, lim_array, sample_points):
    """Plots the probability density histogram of the random walk, with the integrand and weighing function
    accross the integral range.

    Args:
        integrand (function): Defined function that returns function to be integrated 
        weighing_func (function): Defined normalised sampling weighing function
        x (array): Random walk points
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        sample_points (int): Number of runs
    """
    plt.figure()
    x_array = np.linspace(lim_array[0], lim_array[1], sample_points)
    plt.plot(x_array, integrand(x_array), label = 'Integrand function', color = 'orange', linestyle = '--')
    plt.plot(x_array, weighing_func(x_array), label = 'Weighing function', color = 'blue', linestyle = '--')
    plt.hist(x, int(sample_points/10), density = True, label = 'Probability density histogram')
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

def convergence_test(integrand, weighing_func, lim_array, definite_integral, samples, initial_value):
    
    if definite_integral == None:
        definite_integral = importance_sampling_integral(integrand,
                                                         weighing_func,
                                                         lim_array,
                                                         1000,
                                                         initial_value)
        # If no definite integral is known it is estimated with high sample size initially
        
    print("\nRunning convergence test...\n")
    integral = importance_sampling_integral(integrand,
                                            weighing_func,
                                            lim_array,
                                            samples,
                                            initial_value)
    
    print('Initial value = {}'.format(integral[0]))
    while not (99 <= (integral[0]/definite_integral)*100 <= 101):
        # While integral is not ±1% from definite integral defined
        integral = importance_sampling_integral(integrand,
                                                weighing_func,
                                                lim_array,
                                                samples,
                                                initial_value)
        samples += 10 # Calculate integral and increase sample size by 10 each time
    samples01 = samples # Temporary save of sample number, this is the number of 
    # samples it took to get to within ± 1%. Since it increases in steps of 10, this 
    #c an also cross over the next boundary
        
    print("\n ±1'%' accuracy within {} samples...\nvalue = {}\n".format(samples, integral[0]))

    while not (99.5 <= (integral[0]/definite_integral)*100 <= 100.5):
        # Same as before, for new interval
        integral = importance_sampling_integral(integrand,
                                                weighing_func,
                                                lim_array,
                                                samples,
                                                initial_value)
        samples += 1 # Smaller sample size increase for more accurate number
    print("\n ±0.5'%' accuracy within {} samples...\nvalue = {}\n".format(samples, integral[0]))
    return samples01, samples, integral[0] # Return these values for printing

## Below is the routine for checking convergence, commented out to increase code runtime 
# on the above integrals
"""
runs = 10 # Increase runs to increase number of tests
mc_samples_array = np.empty([2,runs])
for b in range(runs):
    values = convergence_test(integrand_1,
                              weighing_func_1,
                              [-10,10],
                              3.544907701811032,
                              10,
                              1) # Run convergence test defined above
    mc_samples_array[0,b] = values[0] # Assign variables to each sample count
    mc_samples_array[1,b] = values[1]
print("\nMean ±1 samples = {}".format(np.mean(mc_samples_array[0,:])))
print("Mean ±0.5 samples = {}".format(np.mean(mc_samples_array[1,:]))) # Find means

mc_samples_array = np.empty([2,runs]) # Same as above, for second integral
for b in range(runs):
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