# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 23:08:01 2018

@author: KaderF
"""
import numpy as np
import matplotlib.pyplot as plt

matterpower = np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float)
plt.xlabel('k')
plt.ylabel('P(k)')
plt.xscale('log')
plt.yscale('log')
plt.plot(matterpower[:,0], matterpower[:,1], color = "black", 
            linestyle = '--', marker = '', label = "matterpower")
plt.show()

#Shows that 1d interpolation works in recovering the matter power spec
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x,y=np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                  unpack=True)
f = interp1d(x, y)
plt.loglog(x,y,'o',x,f(x))

plt.show()

