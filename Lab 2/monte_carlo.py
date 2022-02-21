import math
import numpy as np
import matplotlib.pyplot as plt 

def MonteCarlo_Integrator(func, lim_min, lim_max, sample_points):
    func_vals = np.empty(sample_points)
    var_vals = np.empty_like(func_vals)
    
    samples = (lim_max - lim_min) / sample_points
    vals = np.random.uniform(lim_min, lim_max, sample_points)
    
    for i in range(sample_points):
        func_vals[i] = func(vals[i])
        var_vals[i] = func(vals[i])**2

    integral = samples * np.sum(func_vals)
    variance = np.sqrt( 1/sample_points * ((1/sample_points * np.sum(var_vals)) - (1/sample_points * np.sum(func_vals))))
    
    return integral, variance

def function1(x):
    return 2

def function2(x):
    return -x

def function3(x): 
    return x**2

def function4(x,y):
    return x*y + x

integral1, variance1 = MonteCarlo_Integrator(function1,0,1,10000)
print(integral1, "±", variance1)

integral2, variance2 = MonteCarlo_Integrator(function2,0,1,10000)
print(integral2, "±", variance2)

integral3, variance3 = MonteCarlo_Integrator(function3,-2,2,10000)
print(integral3, "±", variance3)