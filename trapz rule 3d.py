# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:15:18 2018

@author: KaderF
"""
#3D Trapezoidal rule to calculate integral of xyz. You first project onto the xy plane, set z=1 and apply trapz 
#rule once to a 2d array which gives a resultant 1d array. Then multiply resultant 1d array from xy plane with the 
#1d array of z to get another 2d array and then apply trapz rule twice.
import scipy as sp
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

z,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)
M = interp1d(z, P)
H0=70.
m_p_inv=6.25e26

x = np.linspace(1.,1.4e5,np.int(1e5))
#print (x)
y = np.linspace(1.,1.4e5,np.int(1e5))
#print (y)
z=np.linspace(1.e-4,1.0071e+01,np.int(1e5))
#print (z)
res = []
append = res.append
for l in range(200):
    f = x[:,np.newaxis]*y[np.newaxis,:]
    (1.-H0*x[:,np.newaxis]/2.)**(-4)*(1.-H0*y[np.newaxis,:][np.newaxis,:]/2.)**(-4)*np.cos(1.*(x[:,np.newaxis]-y[np.newaxis,:]))/x[:,np.newaxis]**2
    
    #print (f,'first f')
    f=sp.trapz(f, y[np.newaxis,:], axis=1)
    #print (f,'second f')
    f=f[:,np.newaxis]*(z[np.newaxis,:]/(z[np.newaxis,:]**2+(l**2/x**2)))**2*M(z)
    #print (f,'third f')
    ff= (sp.trapz(f, z, axis=1))
    #print (ff)
    fff= (sp.trapz(ff, x, axis=0))
    result=fff*m_p_inv
    append(result)
    
plt.plot(result)
plt.show()
    #print (fff)
#answer is 512 which matches (4^2/2)^3 which is the answer from the integral of int_0^4 int_0^4 int_0^4 xyz