from __future__ import absolute_import
from math import log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from numba import jit

m_sun = 332959
m_J = (1/1000)*m_sun

r1data3bod1 = pd.read_csv("r1data3bod1.csv", delimiter=",")
v1data3bod1 = pd.read_csv("v1data3bod1.csv", delimiter=",")
p1vals = (v1data3bod1.values[:] / 1.6*m_J)

r1data3bod2 = pd.read_csv("r1data3bod2.csv", delimiter=",")
v1data3bod2 = pd.read_csv("v1data3bod2.csv", delimiter=",")
p2vals = (v1data3bod2.values[:] / 1.6*m_J)

tData = pd.read_csv("timedata.csv", delimiter=",")

dt = 1e-4
f = open("lyapunov.csv", "w+")# For debugging, if code failed previously this ensures the data is cleared before another run
f.close()
f = open("dist.csv", "w+")# For debugging, if code failed previously this ensures the data is cleared before another run
f.close()


@jit
def Lyapunov(d1, p1 ,d2, p2, t):
    #d = d1 - d2
    a1 = np.array([d1,p1])
    a2 = np.array([d2,p2])
    diff = np.abs(a1 - a2)
    dist = np.linalg.norm(diff)
    lyp = np.log(dist/d0)/(t)
        
    return lyp
@jit
def Diff(d1, p1 ,d2, p2, t):
    #d = d1 - d2
    a1 = np.array([d1,p1])
    a2 = np.array([d2,p2])
    diff = np.abs(a1 - a2)
    dist = (np.linalg.norm(diff))
        
    return dist

def write(input):  
    with open('lyapunov.csv','a',newline='') as lyps:
            writer = csv.writer(lyps, delimiter=',', lineterminator='\n')
            writer.writerow(input)
def write2(input):  
    with open('dist.csv','a',newline='') as lyps:
            writer = csv.writer(lyps, delimiter=',', lineterminator='\n')
            writer.writerow(input) 


for i in range(len(r1data3bod1.values)-1):
    temp_data = np.zeros(3)
    temp_data_2 = np.zeros(3)
    for j in range(3):
        d0 = np.linalg.norm(np.array(r1data3bod1.values[0,j], p1vals[0,j]) - 
                            np.array(r1data3bod2.values[0,j], p2vals[0,j]))

        temp_data[j] = Lyapunov(r1data3bod1.values[i,j], p1vals[i,j], 
                                r1data3bod2.values[i,j], p2vals[i,j], 
                                tData.values[i+1])
        temp_data_2[j] = Diff(r1data3bod1.values[i,j], p1vals[i,j],
                              r1data3bod2.values[i,j], p2vals[i,j],
                              tData.values[i+1])
    if np.isinf(temp_data[j]) == True:
        temp_data[j] = 0
    write(temp_data)
    if np.isinf(temp_data_2[j]) == True:
        temp_data_2[j] = 0
    write2(temp_data_2)
exit()
    
lypdata = pd.read_csv('lyapunov.csv', delimiter=',')

x_lyps = lypdata.values[0:len(r1data3bod1.values)]
y_lyps = lypdata.values[len(r1data3bod1.values):2*len(r1data3bod1.values)]
z_lyps = lypdata.values[2*len(r1data3bod1.values):3*len(r1data3bod1.values)]

#lyps = nolds.lyap_r(data.values[:,0])
#lyaps2 = nolds.lyap_r(data.values[:,1])
#lyaps3 = nolds.lyap_r(data.values[:,2])

plt.plot(tData[0:len(x_lyps)], x_lyps, label = "$\lambda_x = {}$"
         .format(np.round(np.sum(x_lyps),1)))
plt.plot(tData[0:len(y_lyps)], y_lyps, label = "$\lambda_y = {}$"
         .format(np.round(np.sum(y_lyps),1)))
plt.plot(tData[0:len(z_lyps)], z_lyps, label = "$\lambda_z = {}$"
         .format(np.round(np.sum(z_lyps),1)))
plt.xlabel("Time (Yr)")
plt.ylabel("$\lambda_i$")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()