from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

def importance_sampling_integral(integrand, weighing_func, lim_array, sample_points):

    """ Integrates function using importance sampling, using the metropolis method to take
    nonuniform sample_points of the weighing function.

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

    metropolis_params = [weighing_func, array, lim_array, 10, 3]
    # Initial value defined outside of function,
    # discards first 10 values then takes every 3rd value after that
    while j <= dimensions+(dimensions-1): # Loop through limits via auto counting dimensions

        acceptance = (metropolis_algorithm(*metropolis_params)/sample_points)*100
        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit
        k+=1 # Index just for assigning rand_vals array

    #print('Ratio of values accepted = {}%'.format(int((acceptance/sample_points)*100)))

    function_plots(integrand, weighing_func, array, lim_array, sample_points)

    for i in range(sample_points): # Loop through number of points again
        func_vals += (integrand(*array[i]) / weighing_func(*array[i]))
        var_vals += np.square((integrand(*array[i]) / weighing_func(*array[i])))
        # Determine sum for function value by unpacking each random [xi,yi,zi]
        # Determine sum for variance values similarly

    integral = 1/sample_points * func_vals
    variance = 1/sample_points * (1/sample_points * var_vals -
                                  np.square(1/sample_points * func_vals))
    root_mean_squared = np.sqrt(variance)

    # Collate terms in variance of integration estimation
    end_time = perf_counter()
    time = end_time - start_time

    return integral, time, variance, root_mean_squared, acceptance # Return values

def metropolis_algorithm(sampling_func, array, lim_array,
                         discard_init, discard_sampling):

    """Implementation of the metropolis-hastings algorithm

    Args:
        sampling_func (function): Defined sampling function
        sample_points (int): Number of RUNS
        dimensions (int): Dimensions of integral as defined
        k (int): Global dimension counter, for multidimensional arrays
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        discard_init (int): Number of initial values to discard
        discard_sampling (int): Value defining each (removal_sampling)th
                                value to discard between values

    Returns:
        array (array): Random walk values sampled from the input function
        accepted (float): Number of new trial points accepted and saved during the walk
    """

    accepted = 0
    counter = 0

    iterations = len(array)*discard_sampling + discard_init
    # Number of iterations must be made so that the number of loops still loops through
    # entire array while including the discarded / skipped values
    # (these still count as acceptances regardless of if they are used)

    # More efficient to pre-dedicate memory to the random values rather than generate
    # at each loop:
    initial_value = np.random.uniform(lim_array[j], lim_array[j+1], size = 1)
    sigmas = np.random.normal(0, 1, size = iterations)

    while counter < iterations: # full circle around array

        trial_val = initial_value + sigmas[counter] # create trial value

        frac_val = sampling_func(trial_val) / sampling_func(initial_value)

        if frac_val >= 1 or (frac_val < 1 and np.random.uniform() <= frac_val):
            # acceptance conditions

            initial_value = trial_val
            accepted += 1

        # Discard initial (discard_init) values, and only keep every (discard_sampling)nd value
        # after that
        if (counter > discard_init) and (counter % discard_sampling == 0):
            array[(counter - discard_init) // discard_sampling, k] = initial_value

        counter += 1

    return (accepted/discard_sampling)-discard_init

def function_plots(integrand, weighing_func, x_vals, lim_array, sample_points):

    """Plots the probability density histogram of the random walk,
    with the integrand and weighing function accross the integral range.

    Args:
        integrand (function): Defined function that returns function to be integrated
        weighing_func (function): Defined normalised sampling weighing function
        x (array): Random walk points
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        sample_points (int): Number of RUNS
    """
    plt.figure()
    x_array = np.linspace(lim_array[0], lim_array[1],
                          sample_points)
    plt.plot(x_array, integrand(x_array), label = 'Integrand function',
             color = 'orange', linestyle = '--')
    plt.plot(x_array, weighing_func(x_array), label = 'Weighing function',
             color = 'blue', linestyle = '--')
    plt.hist(x_vals, int(sample_points/10), density = True, label = 'Density histogram')
    plt.xlabel('x')
    plt.xlim(lim_array[0], lim_array[1])
    plt.legend(loc = 'upper right')
    plt.show()

## Convergence correlation with sampling

def convergence_test(integrand, weighing_func, lim_array, definite_integral, sample_points):
    """ Tests convergence rate of the importance_sampling function in terms of number of
    sample_points needed to converge within an error bound of the actual known integral

    Args:
        integrand (function): Defined function that returns function to be integrated
        weighing_func (function): Defined normalised sampling weighing function
        lim_array (array): Array of limits of the form [x_min, x_max, y_min, y_max, ...]
        definite_integral (float): Definite value of the integral to be assessed
        sample_points (int): Number of points to sample

    Returns:
        sample_points01 (int): number of samples needed to converge to ± 1% accuracy
        sample_points (int): number of samples needed to converge to ± 0.1% accuracy
        integral[0] (float): Value returned from importance_sampling for the integral
        integral[1] (float): Time taken for each run
        integral[4] (float): Acceptance rate of metropolis_algorithm in the run
    """

    if definite_integral is None:
        definite_integral = importance_sampling_integral(integrand,
                                                         weighing_func,
                                                         lim_array,
                                                         1000)
        # If no definite integral is known it is estimated with high sample size initially

    integral = importance_sampling_integral(integrand,
                                            weighing_func,
                                            lim_array,
                                            sample_points)
    
    while not 99 <= (integral[0]/definite_integral)*100 <= 101:
        # While integral is not ±1% from definite integral defined
        integral = importance_sampling_integral(integrand,
                                                weighing_func,
                                                lim_array,
                                                sample_points)
        sample_points += 10 # Calculate integral and increase sample size by 10 each time
    sample_points01 = sample_points # Temporary save of sample number, this is the number of
    # sample_points it took to get to within ± 1%. Since it increases in steps of 10, this
    #c an also cross over the next boundary

    while not 99.9 <= (integral[0]/definite_integral)*100 <= 100.1:
        # Same as before, for new interval
        integral = importance_sampling_integral(integrand,
                                                weighing_func,
                                                lim_array,
                                                sample_points)
        sample_points += 1
        # Smaller sample size increase for more accurate number
    return sample_points01, sample_points, integral[0], integral[1], integral[4]
# Return these values for printing

## 5a). approx 3.54

def integrand_1(x_vals):
    """ Function to be integrated as defined

    Args:
        x_vals (array): x values for the function

    Returns:
        float: the function values at each input
    """
    return 2*np.exp(-(x_vals**2))

def weighing_func_1(x_vals):
    """ Weighing function to be sampled as defined

    Args:
        x_vals (array): x values for the function

    Returns:
        float: the function values at each input, normalised via norm_constant
    """
    norm_constant = (1 / (2*(1-np.exp(-10))))
    return  (norm_constant * np.exp(-(np.abs(x_vals))))

integral1, time1, var1, rms1, acc1 = importance_sampling_integral(
                                                integrand_1,
                                                weighing_func_1,
                                                [-10,10],
                                                1000)

print("Integral = {} ± {}, time taken = {}s,\nAcceptance rate = {}%"
      .format(integral1, rms1, time1, acc1))

## 5b). = 3
def integrand_2(x_vals):
    """ Function to be integrated as defined

    Args:
        x_vals (array): x values for the function

    Returns:
        float: the function values at each input
    """
    return 1.5*np.sin(x_vals)

def weighing_func_2(x_vals):
    """ Weighing function to be sampled as defined

    Args:
        x_vals (array): x values for the function

    Returns:
        float: the function values at each input, normalised via norm_constant
    """
    norm_constant = 3 / (2*np.pi)
    return norm_constant * (4/(np.pi**2))*x_vals*(np.pi - x_vals)

integral2, time2, var2, rms2, acc2 = importance_sampling_integral(integrand_2,
                                                weighing_func_2,
                                                [0,np.pi],
                                                1000)

print("Integral = {} ± {}, time taken = {}s,\nAcceptance rate = {}%"
      .format(integral2, rms2, time2, acc2))

exit() # Remove to run below

## Below is the routine for checking convergence, commented out to increase code runtime
# on the above integrals
# NOTE: COMMENT OUT ANY PRINT OR FIGURE PLOTS IN importance_sampling BEFORE RUNNING THE BELOW
# (will show a lot of figures and graphs if not )

RUNS = 10 # Increase RUNS to increase number of tests
information_array = np.empty([5,RUNS])
for b in range(RUNS):
    values = convergence_test(integrand_1,
                              weighing_func_1,
                              [-10,10],
                              3.544907701811032,
                              10) # Run convergence test defined above
    information_array[0,b] = values[0] # Assign variables to each sample count
    information_array[1,b] = values[1]
    information_array[2,b] = values[2] # integral values
    information_array[3,b] = values[3] # time values
    information_array[4,b] = values[4] # acceptance values

print("\nMean ±1 sample_points = {}".format(np.mean(information_array[0,:])))
print("Mean ±0.1 sample_points = {}".format(np.mean(information_array[1,:])))
print("Mean integral value = {}".format(np.mean(information_array[2,:])))
print("Mean time taken = {}s".format(np.mean(np.mean(information_array[3,:]))))
print("Mean acceptance rate = {}".format(np.mean(np.mean(information_array[4,:]))))

information_array = np.empty([5,RUNS]) # Same as above, for second integral
for b in range(RUNS):
    values = convergence_test(integrand_2,
                              weighing_func_2,
                              [0,np.pi],
                              3.0,
                              10)
    information_array[0,b] = values[0] # Assign variables to each sample count
    information_array[1,b] = values[1]
    information_array[2,b] = values[2] # integral values
    information_array[3,b] = values[3] # time values
    information_array[4,b] = values[4] # acceptance values

print("\nMean ±1 sample_points = {}".format(np.mean(information_array[0,:])))
print("Mean ±0.1 sample_points = {}".format(np.mean(information_array[1,:])))
print("Mean integral value = {}".format(np.mean(information_array[2,:])))
print("Mean time taken = {}s".format(np.mean(np.mean(information_array[3,:]))))
print("Mean acceptance rate = {}".format(np.mean(np.mean(information_array[4,:]))))
