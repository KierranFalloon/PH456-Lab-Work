import math
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt

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
        for i in range(sample_points): # Loop through number of points
            rand_vals[i, k] = np.random.uniform(lim_array[j], lim_array[j+1], 1)
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
print(integral1, variance1, rms1, time1)

integral2, variance2, rms2, time2 = montecarlo_integrator(function2,[0,1],10000)
print(integral2, variance2, rms2, time2)

integral3, variance3, rms3, time3 = montecarlo_integrator(function3,[-2,2],10000)
print(integral3, variance3, rms3, time3)

integral4, variance4, rms4, time4 = montecarlo_integrator(function4, [0,1,0,1], 10000)
print(integral4, variance4, rms4, time4)

# Use your subroutine to evaluate the size of the region enclosed
# within an n-sphere of radius 2.0, for n = 3 (i.e. the volume of a ball of
# radius 1.5) and n=5.

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

integral5, variance5, rms5, time5 = montecarlo_integrator(sphere_func_3, [-2,2,-2,2,-2,2], 10000)
integral6, variance6, rms6, time6 = montecarlo_integrator(sphere_func_5, [-2,2,-2,2,-2,2,-2,2,-2,2], 10000)
print(integral5, variance5, rms5, time5)
print(integral6, variance6, rms6, time6)

def function5(ax, ay, az, bx, by, bz, cx, cy, cz):
    a = (ax,ay,az)
    b = (bx,by,bz)
    c = (cx,cy,cz)
    
    return 1 / np.dot(np.add(a,b),c)

integral7, variance7, rms7, time7 = montecarlo_integrator(function5,[0,1,0,1,0,1,
                                                                     0,1,0,1,0,1,
                                                                     0,1,0,1,0,1,], 100000)
print(integral7, variance7, rms7, time7)

def integrand_1(x):
    return np.exp(-(x**2))

def integrand_2(x):
    return 2*np.exp(-(x**2))

def integrand_3(x):
    return 1.5*np.sin(x)

## Convergence correlation with sampling
def convergence_test(integrand, lim_array, definite_integral):
    
    if definite_integral == None:
        definite_integral, a, b, c = montecarlo_integrator(integrand, lim_array, 100000)
        # If no definite integral is known it is estimated with high sample size initially
        
    print("\nRunning convergence test...\n")
    samples = 10 # low initial number
    error = 0.01*definite_integral
    integral = montecarlo_integrator(integrand, lim_array, samples)
    while True:
        print(integral[0])
        print((definite_integral - error) >= integral[0] or integral[0] <= (definite_integral + error))
        if (definite_integral - error) >= integral[0] or integral[0] <= (definite_integral + error):
            integral = montecarlo_integrator(integrand, lim_array, samples)
            samples += 1000
        else:
            integral = montecarlo_integrator(integrand, lim_array, samples)
            return integral[0], samples, integral[2]

#integral, samples, convergence_error = convergence_test(integrand_1, [0,1], 0.746824132812427)
#print(integral, samples, convergence_error)
