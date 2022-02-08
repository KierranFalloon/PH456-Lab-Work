from time import perf_counter
import numpy as np
from numpy.random import Generator, PCG64, MT19937
import matplotlib.pyplot as plt

def chisquare_test(array, bins):

    """Chi-squared test

    Args:
        array: Array to be tested
        bins: Number of bins 'array' is to be split into

    Returns:
        [float]: Chi-squared value
    """

    f_observed = np.histogram(array, bins)[0]  # Split input array into M equal bins
    f_expected = np.ones(bins) * (len(array) / bins)  # Assume equal spacing in bins
    chi_square_array = [0] * bins  # Placeholding empty array
    for i in range(len(array)):
        chi_square_array[i] = (f_observed[i] - f_expected[i]) ** 2 / f_expected[i]
        # chi square equation

    return np.sum(chi_square_array)

############################################
# 
# Comparing two seeds of the same generator
#
############################################

timearr = np.zeros(5)

# Bitgenerator PCG64 (default for numpy), with seed 6
rng = Generator(PCG64(seed=6))

# Pseudo-random, uniform range between 0-1 with defined seed
x1starttime = perf_counter()
x1 = rng.random(2000)
x1endtime = perf_counter()

timearr[0] = x1endtime - x1starttime

x15 = np.roll(x1, 10)  # Move each value of x1 10 spaces along

# Bitgenerator PCG64 (default for numpy), with seed 15832
rng = Generator(PCG64(seed=15832))
# Pseudo-random, uniform range between 0-1 with defined seed
x2starttime = perf_counter()
x2 = rng.random(2000)
x2endtime = perf_counter()
timearr[1] = x2endtime - x2starttime
x25 = np.roll(x2, 10)

####################################
# Statistical tests 
####################################

pcg64chi1 = chisquare_test(x1, 2000)
pcg64chi2 = chisquare_test(x2, 2000)
pcg64chi3 = chisquare_test(x15, 2000)
pcg64chi4 = chisquare_test(x25, 2000)
correl_seeds = np.around(np.correlate(x1, x2)[0], 2)
correl1 = np.around(np.correlate(x1, x15)[0], 2)
correl2 = np.around(np.correlate(x2, x25)[0], 2)
pearsoncoeff_seeds = np.around(np.corrcoef(x1, x2)[0, 1], 5)
pearsoncoeff1 = np.around(np.corrcoef(x1, x15)[0, 1], 5)
pearsoncoeff2 = np.around(np.corrcoef(x2, x25)[0, 1], 5)

####################################
# Plotting routines
####################################
plt.figure(figsize=(6, 6))
plt.scatter(x1, x2, marker="x")
plt.title(
    r"Cross-correlation $\approx %1.2f, \rho_{XY}\approx %1.4f$"
    % (correl_seeds, pearsoncoeff_seeds),
    fontsize=10,)
plt.xlabel("Seed = 6, $\chi^2 = {}$".format(pcg64chi1))
plt.ylabel("Seed = 15832, $\chi^2 = {}$".format(pcg64chi2))
plt.tight_layout()
plt.savefig(fname="PCG64-6-15832")

# Shifted value comparison ( sequential )
plt.figure(figsize=(6, 6))
plt.scatter(x1, x15, marker="x")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n+10}$")
plt.title(
    r"Cross-correlation $\approx %1.2f$, $\chi^2 \approx %s, \rho_{XY}\approx %1.4f$"
    % (correl1, pcg64chi1, pearsoncoeff1),
    fontsize=10,
)
plt.tight_layout()
plt.savefig(fname="PCG64-6")

# Shifted value comparison ( sequential )
plt.figure(figsize=(6, 6))
plt.scatter(x2, x25, marker="x")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n+10}$")
plt.title(
    r"Cross-correlation $\approx %1.2f$, $\chi^2 \approx %s, \rho_{XY}\approx %1.4f$"
    % (correl2, pcg64chi2, pearsoncoeff2),
    fontsize=10,
)
plt.tight_layout()
plt.savefig(fname="PCG64-15832")


#####################################################
# 
# Different Generator - Same seed Comparison
#
#####################################################

# Bitgenerator MT19937 (default for numpy), with seed 6
rng = Generator(MT19937(6))
# Pseudo-random, uniform range between 0-1 with defined seed
y1starttime = perf_counter()
y1 = rng.random(2000)
y1endtime = perf_counter()
timearr[2] = y1endtime - y1starttime


####################################
# Statistical tests 
####################################

mt19937chi1 = chisquare_test(y1, 2000)
correl_seeds2 = np.around(np.correlate(x1, y1)[0], 2)
pearsoncoeff_seeds2 = np.around(np.corrcoef(x1, y1)[0, 1], 5)

####################################
# Plotting routine
####################################

plt.figure(figsize=(6, 6))
plt.scatter(x1, y1, marker="x")
plt.title(
    r"Cross-correlation $\approx %1.2f, \rho_{XY}\approx %1.4f$"
    % (correl_seeds2, pearsoncoeff_seeds2),
    fontsize=10,)
plt.xlabel("PCG64, seed = 6, $\chi^2 = {}$".format(pcg64chi1))
plt.ylabel("MT19937, seed = 6, $\chi^2 = {}$".format(mt19937chi1))
plt.tight_layout()
plt.savefig(fname="PCG64-MT19937-6")

#####################################################
# 
# Same Generator MT19937 - Different seed Comparison
#
#####################################################

# Bitgenerator MT19937 (default for numpy), with seed 5525
rng = Generator(MT19937(5255))

# Pseudo-random, uniform range between 0-1 with defined seed
y1starttime = perf_counter()
y1 = rng.random(2000)
y1endtime = perf_counter()
timearr[3] = y1endtime - y1starttime
y15 = np.roll(y1, 10)

# Bitgenerator MT19937 (default for numpy), with seed 7381
rng = Generator(MT19937(7381))

# Pseudo-random, uniform range between 0-1 with defined seed
y2starttime = perf_counter()
y2 = rng.random(2000)
y2endtime = perf_counter()
timearr[4] = y2endtime - y2starttime
y25 = np.roll(y2, 10)

####################################
# Statistical tests 
####################################
mt19937chi1 = chisquare_test(y1, 2000)
mt19937chi2 = chisquare_test(y2, 2000)
mt19937chi3 = chisquare_test(y15, 2000)
mt19937chi4 = chisquare_test(y25, 2000)
correl_seeds3 = np.around(np.correlate(y1, y2)[0], 2)
correl3 = np.around(np.correlate(y1, y15)[0], 2)
correl4 = np.around(np.correlate(y2, y25)[0], 2)
pearsoncoeff_seeds3 = np.around(np.corrcoef(y1, y2)[0, 1], 5)
pearsoncoeff3 = np.around(np.corrcoef(y1, y15)[0, 1], 5)
pearsoncoeff4 = np.around(np.corrcoef(y2, y25)[0, 1], 5)

####################################
# Plotting routine
####################################

plt.figure(figsize=(6, 6))
plt.scatter(y1, y2, marker="x")
plt.title(
    r"Cross-correlation $\approx %1.2f, \rho_{XY}\approx %1.4f$"
    % (correl_seeds3, pearsoncoeff_seeds3),
    fontsize=10,)
plt.xlabel("Seed = 5255, $\chi^2 = {}$".format(mt19937chi1))
plt.ylabel("Seed = 7381, $\chi^2 = {}$".format(mt19937chi2))
plt.tight_layout()
plt.savefig(fname="MT19937-5525-7381")

################################################
# Plotting routine
################################################

plt.figure(figsize=(6, 6))
plt.scatter(y1, y15, marker="x")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n+10}$")
plt.title(
    r"Cross-correlation $\approx %1.2f$, $\chi^2 \approx %s, \rho_{XY}\approx %1.4f$"
    % (correl3, mt19937chi1, pearsoncoeff3),
    fontsize=10,
)
plt.tight_layout()
plt.savefig(fname="MT19937-5255")


plt.figure(figsize=(6, 6))
plt.scatter(y2, y25, marker="x")
plt.xlabel("$x_n$")
plt.ylabel("$x_{n+10}$")
plt.title(
    r"Cross-correlation $\approx %1.2f$, $\chi^2 \approx %s, \rho_{XY}\approx %1.4f$"
    % (correl4, mt19937chi2, pearsoncoeff4),
    fontsize=10,
)
plt.tight_layout()
plt.savefig(fname="MT19937-7381")

################################################
# Time taken for each random generator
################################################

print(
    "\nTimes: \n PCG64-6 =",
    timearr[0] / 1e-6,
    "µs\n",
    "PCG64-15832 =",
    timearr[1] / 1e-6,
    "µs\n",
    "MT19937-6 =",
    timearr[2] / 1e-6,
    "µs\n",
    "MT19937-5255 =",
    timearr[3] / 1e-6,
    "µs\n",
    "MT19937-7381 =",
    timearr[4] / 1e-6,
    "µs\n",
)

################################################
# Comparing shifted value vs np.correlate values
################################################

shift = np.arange(0, 1000, 1)

rng = Generator(PCG64(2010))
# Pseudo-random, uniform range between 0-1 with defined seed
correlation1 = np.zeros(1000)
pcg642010chi = np.zeros_like(correlation1)
z1 = rng.random(1000)
pearsoncoeff5 = np.zeros_like(correlation1)
for i in range(1000):
    z15 = np.roll(z1, shift[i])
    correlation1[i] = np.correlate(z1, z15)
    pcg642010chi[i] = chisquare_test(z15, 1000)
    pearsoncoeff5[i] = np.around(np.corrcoef(z1, z15)[0, 1], 5)

rng = Generator(MT19937(2010))
# Pseudo-random, uniform range between 0-1 with defined seed
correlation2 = np.zeros(1000)
mt199372010chi = np.zeros_like(correlation2)
pearsoncoeff6 = np.zeros_like(correlation2)
z2 = rng.random(1000)

for i in range(1000):
    z25 = np.roll(z2, shift[i])
    correlation2[i] = np.correlate(z2, z25)
    mt199372010chi[i] = chisquare_test(z25, 1000)
    pearsoncoeff6[i] = np.around(np.corrcoef(z2, z25)[0, 1], 5)

# Values for visualisation
range1 = np.round(correlation1[1:1000].max() - correlation1.min(), 2)
range2 = np.round(correlation2[1:1000].max() - correlation2.min(), 2)
std1 = np.round(np.std(correlation1[1:1000]), 2)
std2 = np.round(np.std(correlation2[1:1000]), 2)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(shift, correlation1)
ax1.set_title("PCG64")
ax1.text(shift[5], correlation1.max() - 5, "Max = {}".format(np.round(correlation1.max(), 2)))
ax1.hlines(np.mean(correlation1[1:1000]),0,1000,"r","--",
           label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
           .format(np.round(np.mean(correlation1[1:1000]), 2), std1, range1))

ax2.plot(shift, correlation2)
ax2.set_title("MT19937")
ax2.text(shift[5], correlation2.max() - 5, "Max = {}".format(np.round(correlation2.max(), 2)))
ax2.hlines(np.mean(correlation2[1:1000]),0,1000,"r","-.",
           label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
           .format(np.round(np.mean(correlation2[1:1000]), 2), std2, range2))
ax2.set_xlabel("Shifted index $i$")

fig.supylabel("Cross-correlation between $x_n~&~x_{n+i}$")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig("Correlation-shift-comparison")

################################################
# Comparing shifted value vs pearson values
################################################

pearson_range1 = np.round(pearsoncoeff5[1:1000].max() - pearsoncoeff5.min(), 4)
pearson_range2 = np.round(pearsoncoeff6[1:1000].max() - pearsoncoeff6.min(), 4)
pearson_std1 = np.round(np.std(pearsoncoeff5[1:1000]), 4)
pearson_std2 = np.round(np.std(pearsoncoeff6[1:1000]), 4)

fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(shift, pearsoncoeff5)
ax1.set_title("PCG64")
ax1.hlines(np.mean(pearsoncoeff5[1:1000]),0,1000,"r","--",
           label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
           .format(np.round(np.mean(pearsoncoeff5[1:1000])*-1, 2), pearson_std1, pearson_range1))

ax2.plot(shift, pearsoncoeff6)
ax2.set_title("MT19937")
ax2.hlines(np.mean(pearsoncoeff6[1:1000]),0,1000,"r","--",
           label="$\mu = {}$,\n$\sigma = {}$,\n$\Delta = {}$"
           .format(np.round(np.mean(pearsoncoeff6[1:1000])*-1, 2), pearson_std2, pearson_range2))

ax2.set_xlabel("Shifted index $i$")
fig2.supylabel(r"$\rho_{XY}$ between $x_n~&~x_{n+i}$")
ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig("Pearson-shift-comparison")
