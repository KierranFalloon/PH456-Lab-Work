import math
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def montecarlo_integrator(func, lim_array, sample_points):

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
    rand_vals = np.empty((sample_points, dimensions)) 
    # Empty random vals matrix size (sample_points rows, dimensions columns)

    j=0 # Dimensions counter
    k=0 # Another counter for indexing
    area = 1 # area initial value

    while j <= dimensions+(dimensions-1): # Loop through limits via auto counting dimensions
        area *= lim_array[j+1] - lim_array[j]
        # area = Previous difference in limits * this one, for all limits
        rand_vals[:, k] = np.random.uniform(lim_array[j], lim_array[j+1], sample_points)
        # Create random number and store it in the array, such that each random number
        # is within the input limits
        j+=2 # Always of form [lower, upper, lower, upper, ...] so +2 to get to next lower limit
        k+=1 # Index just for assigning rand_vals array

    for i in range(sample_points): # Loop through number of points again
        func_vals += func(*rand_vals[i])
        # Determine sum for function value by unpacking each random [xi,yi,zi]
        var_vals += np.square(func(*rand_vals[i]))
        # Determine sum for variance values similarly

    integral = area/sample_points * func_vals 
    variance = 1/sample_points * (1/sample_points * var_vals - 
                                  np.square(1/sample_points * func_vals))
    
    # Collate terms in integration estimation
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

integral1, variance1, rms1, time1 = montecarlo_integrator(function1,[0,1],10000)
print("Integral = {} ± {}, time taken = {}s".format(integral1, rms1, time1))

integral2, variance2, rms2, time2 = montecarlo_integrator(function2,[0,1],10000)
print("Integral = {} ± {}, time taken = {}s".format(integral2, rms2, time2))

integral3, variance3, rms3, time3 = montecarlo_integrator(function3,[-2,2],10000)
print("Integral = {} ± {}, time taken = {}s".format(integral3, rms3, time3))

integral4, variance4, rms4, time4 = montecarlo_integrator(function4, [0,1,0,1], 10000)
print("Integral = {} ± {}, time taken = {}s".format(integral4, rms4, time4))

# Use your subroutine to evaluate the size of the region enclosed
# within an n-sphere of radius 2.0, for n = 3 (i.e. the volume of a ball of
# radius 1.5) and n=5.

def sphere_func(*coordinates):
    
    coordinates_array = np.array(coordinates)
    if np.linalg.norm(coordinates_array) <= 2:
        return 1
    else:
        return 0


integral5, variance5, rms5, time5 = montecarlo_integrator(sphere_func, [-2,2,-2,2,-2,2], 10000)
print("Integral = {} ± {}, time taken = {}s".format(integral5, rms5, time5))
integral6, variance6, rms6, time6 = montecarlo_integrator(sphere_func, [-2,2,-2,2,-2,2,-2,2,-2,2], 10000)
print("Integral = {} ± {}, time taken = {}s".format(integral6, rms6, time6))

def function5(*coordinates):
    a = (coordinates[0], coordinates[1], coordinates[2])
    b = (coordinates[3], coordinates[4], coordinates[5])
    c = (coordinates[6], coordinates[7], coordinates[8])
    
    return 1 / np.dot(np.add(a,b),c)

integral7, variance7, rms7, time7 = montecarlo_integrator(function5,[0,1,0,1,0,1,
                                                                     0,1,0,1,0,1,
                                                                     0,1,0,1,0,1,], 100000)
print("Integral = {} ± {}, time taken = {}s".format(integral7, rms7, time7))

def integrand_1(x):
    return 2*np.exp(-(x**2))

def integrand_2(x):
    return 1.5*np.sin(x)

## Convergence correlation with sampling
def convergence_test(integrand, lim_array, definite_integral, samples):
    
    if definite_integral == None:
        definite_integral, a, b, c = montecarlo_integrator(integrand, lim_array, 100000)
        # If no definite integral is known it is estimated with high sample size initially
        
    #print("\nRunning convergence test...\n")
    integral = montecarlo_integrator(integrand, lim_array, samples)
    #print('Initial value = {}'.format(integral[0]))
    while not (99 <= (integral[0]/definite_integral)*100 <= 100):
        #print((integral[0]/definite_integral)*100)
        integral = montecarlo_integrator(integrand, lim_array, samples)
        samples += 100
        
    samples1 = samples
        
    #print("\n ±1'%' accuracy within {} samples...\nvalue = {}\n".format(samples, integral[0]))

    while not (99.9 <= (integral[0]/definite_integral)*100 <= 100.1):
        #print((integral[0]/definite_integral)*100)
        integral = montecarlo_integrator(integrand, lim_array, samples)
        samples += 1
    
    #print("\n ±0.5'%' accuracy within {} samples...\nvalue = {}\n".format(samples, integral[0]))
    
    return samples1, samples, integral[0], integral[3]


runs = 10 # Increase runs to increase number of tests
information_array = np.empty([4,runs])
for b in range(runs):
    values = convergence_test(integrand_1,
                              [-10,10],
                              3.544907701811032,
                              10) # Run convergence test defined above
    information_array[0,b] = values[0] # Assign variables to each sample count
    information_array[1,b] = values[1]
    information_array[2,b] = values[2] # integral values
    information_array[3,b] = values[3] # time values

print("\nMean ±1 sample_points = {}".format(np.mean(information_array[0,:])))
print("Mean ±0.1 sample_points = {}".format(np.mean(information_array[1,:])))
print("Mean integral value = {}".format(np.mean(information_array[2,:])))
print("Mean time taken = {}s".format(np.mean(np.mean(information_array[3,:]))))

information_array = np.empty([4,runs]) # Same as above, for second integral
for b in range(runs):
    values = convergence_test(integrand_2,
                              [0,np.pi],
                              3.0,
                              10)
    information_array[0,b] = values[0] # Assign variables to each sample count
    information_array[1,b] = values[1]
    information_array[2,b] = values[2] # integral values
    information_array[3,b] = values[3] # time values

print("\nMean ±1 sample_points = {}".format(np.mean(information_array[0,:])))
print("Mean ±0.1 sample_points = {}".format(np.mean(information_array[1,:])))
print("Mean integral value = {}".format(np.mean(information_array[2,:])))
print("Mean time taken = {}s".format(np.mean(np.mean(information_array[3,:]))))

