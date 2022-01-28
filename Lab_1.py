"""
    The aims the exercise this week is to produce and test random numbers
    and then apply them to a simple partitioned box problem, similar to the
    case discussed in class.

"""

import numpy as np
from numpy.random import Generator, PCG64, MT19937
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.stats import chisquare

def chisquare_test(array, bins):
    f_observed, edges = np.histogram(array, bins) # Split input array into M equal bins
    f_expected = np.ones(bins)*( len(array) / bins ) # Assume equal spacing in bins
    chi_square_array = [0]*bins # Placeholding empty array
    for i in range(len(array)):
        chi_square_array[i] = ( (f_observed[i] - f_expected[i])**2 / f_expected[i]) # chi square equation
    
    return np.sum(chi_square_array)

# --------
# Using different initial seeds and at least two different pseudo-random
# generators, produce sequences of uniformly distributed random numbers.

# Test these values for
# a) uniformity and
# b) lack of sequential correlation.
# Present your analysis as graphically as possible.
# --------

# -------- Using two different seeds on the same generator should result in reproducible,
#  but different sequences of values

timearr = np.zeros(5)

# Bitgenerator PCG64 (default for numpy), with seed 6
rng = Generator(PCG64(seed=6))
print(rng)  # Bitgenerator check
# Pseudo-random, uniform range between 0-1 with defined seed
x1starttime = perf_counter()
x1 = rng.uniform(0, 1, 100)
x1endtime = perf_counter()
timearr[0] = x1endtime-x1starttime
x15 = np.roll(x1, 10)  # Move each value of x1 10 spaces along
pcg64chi1 = chisquare_test(x1,100)

# Bitgenerator PCG64 (default for numpy), with seed 15832
rng = Generator(PCG64(seed=15832))
print(rng)
# Pseudo-random, uniform range between 0-1 with defined seed
x2starttime = perf_counter()
x2 = rng.uniform(0, 1, 100)
x2endtime = perf_counter()
timearr[1] = x2endtime-x2starttime
x25 = np.roll(x2, 10)
pcg64chi2 = chisquare_test(x2,100)

# Seed Comparison
plt.figure(figsize=(6, 6))
plt.scatter(x1, x2, marker='x')
plt.xlabel('Seed = 6, $\u03C7^2 = {}$'.format(pcg64chi1))
plt.ylabel('Seed = 15832, $\u03C7^2 = {}$'.format(pcg64chi2))
plt.title('PCG64\n')
plt.tight_layout()
plt.savefig(fname='IMAGES/PCG64-6-15832')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(x1, x15, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('PCG64, seed = 6\n Autocorrelation = {}, $\u03C7^2 = {}$'.format(np.correlate(x1, x15), pcg64chi1))
plt.tight_layout()
plt.savefig(fname='IMAGES/PCG64-6')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(x2, x25, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('PCG64, seed = 15832\n Autocorrelation = {}, $\u03C7^2 = {}$'.format(np.correlate(x2, x25), pcg64chi2))
plt.tight_layout()
plt.savefig(fname='IMAGES/PCG64-15832')
plt.show()


# -------- Using the same seeds on two different generators should result in reproducible,
#  but different sequences of values

# Bitgenerator MT19937 (default for numpy), with seed 6
rng = Generator(MT19937(6))
print(rng)
# Pseudo-random, uniform range between 0-1 with defined seed
y1starttime = perf_counter()
y1 = rng.uniform(0, 1, 100)
y1endtime = perf_counter()
timearr[2] = y1endtime-y1starttime
mt19937chi1 = chisquare_test(y1,100)

# Different Generator - Same seed Comparison
plt.figure(figsize=(6, 6))
plt.scatter(x1, y1, marker='x')
plt.xlabel('PCG64, seed = 6, $\u03C7^2 = {}$'.format(pcg64chi1))
plt.ylabel('MT19937, seed = 6, $\u03C7^2 = {}$'.format(mt19937chi1))
plt.title('PCG64-MT19937-6')
plt.tight_layout()
plt.savefig(fname='IMAGES/PCG64-MT19937-6')
plt.show()

# Bitgenerator MT19937 (default for numpy), with seed 5525
rng = Generator(MT19937(5255))
# Pseudo-random, uniform range between 0-1 with defined seed
y1starttime = perf_counter()
y1 = rng.uniform(0, 1, 100)
y1endtime = perf_counter()
timearr[3]=y1endtime-y1starttime
y15 = np.roll(y1, 10)
mt19937chi1 = chisquare_test(y1,100)

# Bitgenerator MT19937 (default for numpy), with seed 7381
rng = Generator(MT19937(7381))
# Pseudo-random, uniform range between 0-1 with defined seed
y2starttime = perf_counter()
y2 = rng.uniform(0, 1, 100)
y2endtime = perf_counter()
timearr[4] = y2endtime-y2starttime
y25 = np.roll(y2, 10)
mt19937chi2 = chisquare_test(y2,100)

# Seed Comparison
plt.figure(figsize=(6, 6))
plt.scatter(y1, y2, marker='x')
plt.xlabel('Seed = 5255, $\u03C7^2 = {}$'.format(mt19937chi1))
plt.ylabel('Seed = 7381, $\u03C7^2 = {}$'.format(mt19937chi2))
plt.title('MT19937\n')
plt.tight_layout()
plt.savefig(fname='IMAGES/MT19937-5525-7381')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(y1, y15, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('MT19937, seed = 5255\n Autocorrelation = {}, $\u03C7^2 = {}$'.format(np.correlate(y1, y15), mt19937chi1))
plt.tight_layout()
plt.savefig(fname='IMAGES/MT19937-5255')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(y2, y25, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('MT19937, seed = 7381\n Autocorrelation = {}, $\u03C7^2 = {}$'.format(np.correlate(y2, y25), mt19937chi2))
plt.tight_layout()
plt.savefig(fname='IMAGES/MT19937-7381')
plt.show()

print('\nTimes: \n PCG64-6 =', timearr[0]/1e-6,'µs\n', 'PCG64-15832 =', timearr[1]/1e-6,'µs\n', 'MT19937-6 =', timearr[2]/1e-6,'µs\n', 'MT19937-5255 =', timearr[3]/1e-6,'µs\n' ,'MT19937-7381 =', timearr[4]/1e-6,'µs\n')

