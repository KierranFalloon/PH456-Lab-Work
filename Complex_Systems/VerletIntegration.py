import math
import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

f = open("r1data3bod1.csv", "w+")# For debugging, if code failed previously this ensures the data is cleared before another run
f.close()
f = open("v1data3bod1.csv", "w+")# For debugging, if code failed previously this ensures the data is cleared before another run
f.close()
f = open("r2data3bod1.csv", "w+")
f.close()
f = open("v2data3bod1.csv", "w+")# For debugging, if code failed previously this ensures the data is cleared before another ru
f.close()
f = open("r3data3bod1.csv", "w+")
f.close()
f = open("v3data3bod1.csv", "w+")# For debugging, if code failed previously this ensures the data is cleared before another ru
f.close()
f = open("timedata.csv","w+")
f.close()

# number of iteration steps
dt = 1e-4 # timestep size
output_stride = 100 # stride for output to disk
# outputfilename

Natom = 3 # number of atoms

# allocate global arrays for position,
# box counter (Task5), velocity, force and mass


#Defining the Gravitational Constant 
G = 1.185684e-4 #in the system whose base units are Astronomical Units, Masses of the Earth, and years.
m_sun = 332959
m_J = (1/1000)*m_sun

# Gamma Cephei circumbinary orbit data
m1 = 1.6*m_J # Ab mass
m2 = 1.4*m_sun # A mass
m3 = 0.362*m_sun # B mass
totmass = m1 + m2 +m3

#Defining positions of the two atoms, so that barycentre (C.o.M) is at origin
r1 = np.zeros(3)
r2 = np.copy(r1)
r3 = np.copy(r1)

r1[0] = 6.2 # Ab
#r1[1] = -0.5
#r1[2] = -0.5

r2[0] = 4 # A
r3 = -(m1*r1 + m2*r2)/m3 # B: Distance approx. 19AU as defined, centered on 0

v1 = np.zeros(3)
v2 = np.copy(v1)
v3 = np.copy(v1)

v1[1] = 6 # Arb. velocities such to emulate an orbit
v2[1] = 0.3
v3 = -(m1*v1 + m2*v2)/m3

com = (((1 / (m2+m3))*((m2*r2) + (m3*r3))))
comv = (((1 / (m1+m2+m3))*((m1*v1) + (m2*v2) + (m3*v3))))

v = [v1,v2,v3]
r = [r1,r2,r3] # Changing to vectors
m = [m1,m2,m3]


print('\n\nModelling; \n\n Number of atoms = {}\n\n'.format(Natom),'Positions\n r1 = {}\n'.format(r1), 'r2 = {}\n'.format(r2),'r3 = {}\n'.format(r3),'|r2 - r3| = {}\n\n'.format(np.abs(r2-r3)),'Velocities\n v1 = {}\n'.format(v1), 'v2 = {}\n'.format(v2),'v3 = {}\n\n'.format(v3),\
    'Mass of bodies \n m1 = {}\n'.format(m1),'m2 = {}\n'.format(m2),'m3 = {}\n\n'.format(m3),'C.o.M = {}\n'.format(com),'V-C.o.M = {}\n\n'.format(comv),'Timestep = {}'.format(dt)) # Debugging

@jit
def GravForce(G,m,r):

        forces = [0,0,0]
        for i in range(3): # Number of masses 
                for j in range(i+1,3): # loops through other masses, discounting i th mass
                        dist = r[i]-r[j] # Distance between i and j th body, in 3 dimensions
                        F = -G*m[i]*m[j]*dist/(np.linalg.norm(dist))**3 # Force
                        F2 = -F
                        forces[i] += F/m[i] # Forces on i th body
                        forces[j] += F2/m[j] # Forces due to i th body on j th bodies
        return np.array(forces) # Change to a numpy array


    # F = - GmMr/r^2
############################################################

def write(r, v, dt): 

    if (i%output_stride == 0):
            
        with open('r1data3bod1.csv','a',newline='') as P1Pos:
                writer = csv.writer(P1Pos, delimiter=',', lineterminator='\n')
                writer.writerow(r[0])      
        with open('r2data3bod1.csv','a',newline='') as P2Pos:
                writer = csv.writer(P2Pos, delimiter=',', lineterminator='\n')
                writer.writerow(r[1])
        with open('r3data3bod1.csv','a',newline='') as P3Pos:
                writer = csv.writer(P3Pos, delimiter=',', lineterminator='\n')
                writer.writerow(r[2])

        with open('v1data3bod1.csv','a',newline='') as P1Vel:
                writer = csv.writer(P1Vel, delimiter=',', lineterminator='\n')
                writer.writerow(v[0]) 
        with open('v2data3bod1.csv','a',newline='') as P2Vel:
                writer = csv.writer(P2Vel, delimiter=',', lineterminator='\n')
                writer.writerow(v[1]) 
        with open('v3data3bod1.csv','a',newline='') as P3Vel:
                writer = csv.writer(P3Vel, delimiter=',', lineterminator='\n')
                writer.writerow(v[2]) 

        with open('timedata.csv','a',newline='') as times:
                writer = csv.writer(times, delimiter=',', lineterminator='\n')
                writer.writerow([dt*i])              
                

def Verlet(r,v,m,runs):
        F = GravForce(G,m,r)
        global i
        i = 0
        
        while i < runs:
        #Save the current positions
                write(r, v, dt) # Write (initial) data to the file as defined earlier

                # Calculate half step velocity as a list of vectors, v = [ [vx1,vy1,vz1], [vx2,vy2,vz2], ...]
                v_half = v + (dt/2)*F
                # Calculate position at full step using half step
                r_full = r + dt*v_half 

                F = GravForce(G,m,r_full)
                # Update velocity at full step using full step position
                v_full = v_half + (dt/2)*F

                r = r_full
                v = v_full
                
                i+=1
        

runs = 10000000
start = time.time()
Verlet(r, v, m, runs)
end = time.time()


"""
f = open("r1data3bod1.csv", "w+")
f.close()
f = open("r2data3bod1.csv", "w+") 
f.close()
f = open("r3data3bod1.csv", "w+") 
f.close()
f = open("timedata.csv","w+")
"""

print('Time taken = {}s'.format(end-start))