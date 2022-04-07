import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from scipy import stats

def orbit():
    r1data3bod2 = np.genfromtxt('r1data3bod2.csv', delimiter=',')
    r2data3bod2 = np.genfromtxt('r2data3bod2.csv', delimiter=',')
    r3data3bod2 = np.genfromtxt('r3data3bod2.csv', delimiter=',')

    v1data3bod2 = np.genfromtxt('v1data3bod2.csv', delimiter=',')
    v2data3bod2 = np.genfromtxt('v2data3bod2.csv', delimiter=',')
    v3Data3bod2 = np.genfromtxt('v3data3bod2.csv', delimiter=',')


    plt.figure(figsize = (8,6))
    plt.plot(r1data3bod2[1:,0],r1data3bod2[1:,1], label = 'Ab $1.6M_J$')
    plt.plot(r2data3bod2[1:,0],r2data3bod2[1:,1], label = 'A $1.4M_{\odot}$')
    plt.plot(r3data3bod2[1:,0],r3data3bod2[1:,1], label = 'B $0.362M_{\odot}$')
    plt.xlabel('x position (AU)')
    plt.ylabel('y position (AU)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def lyp():
    lypdata = pd.read_csv('lyapunov.csv', delimiter=',')
    t = np.arange(0,len(lypdata), 1)
    #res3 = stats.linregress(np.log10(t), np.log10(lypdata.values[:,2]))
    
    plt.plot(np.log10(t), np.log10(lypdata.values[:,0]), label = 'x')
    plt.plot(np.log10(t), np.log10(lypdata.values[:,1]), label = 'y')
    plt.plot(np.log10(t), np.log10(lypdata.values[:,2]), label = 'z')
    plt.xlabel("log$_{10}$ Time (years)")
    plt.ylabel("log$_{10} \gamma$ (years$^{-1}$)")
    plt.grid(True)
    plt.xlim(1,5)
    plt.legend()
    plt.show()

    
def dists():
    distdata = pd.read_csv('dist.csv', delimiter=',')
    t = np.arange(0,len(distdata[:]), 1)
    t_2 = t[70000:]
    
    res1 = stats.linregress(t_2, distdata.values[70000:,0])
    res2 = stats.linregress(t_2, distdata.values[70000:,1])
    res3 = stats.linregress(t_2, distdata.values[70000:,2])
    plt.figure()
    plt.plot(np.log10(t_2), np.log10(distdata.values[70000:,0]), label = 'x, m = {}'.format(np.round(res1.slope, 2)))
    plt.plot(np.log10(t_2), np.log10(distdata.values[70000:,1]), label = 'y, m = {}'.format(np.round(res2.slope,1)))
    plt.plot(np.log10(t_2), np.log10(distdata.values[70000:,2]), label = 'z, m = {}'.format(res3.slope))
    plt.xlabel("log$_{10}$ Time (years)")
    plt.legend()
    plt.show()

#orbit()
lyp()
dists()