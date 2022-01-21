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

# Bitgenerator PCG64 (default for numpy), with seed 15832
rng = Generator(PCG64(seed=15832))
print(rng)
# Pseudo-random, uniform range between 0-1 with defined seed
x2starttime = perf_counter()
x2 = rng.uniform(0, 1, 100)
x2endtime = perf_counter()
timearr[1] = x2endtime-x2starttime
x25 = np.roll(x2, 10)

# Seed Comparison
plt.figure(figsize=(6, 6))
plt.scatter(x1, x2, marker='x')
plt.xlabel('Seed = 6')
plt.ylabel('Seed = 15832')
plt.title('PCG64')
plt.tight_layout()
plt.savefig(fname='PCG64-6-15832')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(x1, x15, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('PCG64, seed = 6\n Autocorrelation = {}'.format(
    np.correlate(x1, x15)))
plt.tight_layout()
plt.savefig(fname='PCG64-6')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(x2, x25, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('PCG64, seed = 15832\n Autocorrelation = {}'.format(
    np.correlate(x2, x25)))
plt.tight_layout()
plt.savefig(fname='PCG64-15832')
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

# Different Generator - Same seed Comparison
plt.figure(figsize=(6, 6))
plt.scatter(x1, y1, marker='x')
plt.xlabel('PCG64, seed = 6')
plt.ylabel('MT19937, seed = 6')
plt.title('PCG64-MT19937-6')
plt.tight_layout()
plt.savefig(fname='PCG64-MT19937-6')
plt.show()

# Bitgenerator MT19937 (default for numpy), with seed 5525
rng = Generator(MT19937(5255))
# Pseudo-random, uniform range between 0-1 with defined seed
y1starttime = perf_counter()
y1 = rng.uniform(0, 1, 100)
y1endtime = perf_counter()
timearr[3]=y1endtime-y1starttime
y15 = np.roll(y1, 10)

# Bitgenerator MT19937 (default for numpy), with seed 7381
rng = Generator(MT19937(7381))
# Pseudo-random, uniform range between 0-1 with defined seed
y2starttime = perf_counter()
y2 = rng.uniform(0, 1, 100)
y2endtime = perf_counter()
timearr[4] = y2endtime-y2starttime
y25 = np.roll(y2, 10)

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(y1, y15, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('MT19937, seed = 5255\n Autocorrelation = {}'.format(
    np.correlate(y1, y15)))
plt.tight_layout()
plt.savefig(fname='MT19937-5255')
plt.show()

# Shifted value comparison ( uniformity )
plt.figure(figsize=(6, 6))
plt.scatter(y2, y25, marker='x')
plt.xlabel('$x_n$')
plt.ylabel('$x_{n+10}$')
plt.title('MT19937, seed = 7381\n Autocorrelation = {}'.format(
    np.correlate(y2, y25)))
plt.tight_layout()
plt.savefig(fname='MT19937-7381')
plt.show()

print('\nTimes: \n PCG64-6 =', timearr[0]/1e-6,'µs\n', 'PCG64-15832 =', timearr[1]/1e-6,'µs\n', 'MT19937-6 =', timearr[2]/1e-6,'µs\n', 'MT19937-5255 =', timearr[3]/1e-6,'µs\n' ,'MT19937-7381 =', timearr[4]/1e-6,'µs\n')
