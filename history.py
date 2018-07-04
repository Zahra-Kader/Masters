# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Thu Apr 19 16:04:57 2018)---
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
%clear
unix_times = data_file['index_map/time']['ctime']
print(unix_times)


# We can convert these to a more convenient format using astropy:
times = Time(unix_times, format='unix')
iso_times = times.iso
print(iso_times)


# This gives us the timestamps in UTC in ISO format.
# Additionally we can look at the data index in terms of frequency bands:
%clear
data_file = h5py.File('../Downloads/00329962_0000.h5', 'r')


# This file was taken between about midnight and 3AM SAST on October 16th.
# Looking at the index map, we can print out the members of this group with:
print(list(data_file['index_map'].keys()))


# Here we can find the timestamps in ctime format using:
unix_times = data_file['index_map/time']['ctime']
print(unix_times)


# We can convert these to a more convenient format using astropy:
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
90-72.9

90.0-72.9
%clear
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/hirax_tools_usage.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/hirax_tools_usage.py', wdir='C:/Users/kaderf/Downloads')

## ---(Mon Apr 23 09:32:23 2018)---
runfile('C:/Users/kaderf/Downloads/hirax_intro_notebook.py', wdir='C:/Users/kaderf/Downloads')

## ---(Mon May 28 13:37:05 2018)---
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')
%clear
runfile('C:/Users/kaderf/Downloads/integral_trial.py', wdir='C:/Users/kaderf/Downloads')

## ---(Sun Jun  3 15:27:19 2018)---
runfile('C:/Users/kaderf/Downloads/script_mps.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/matterps.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/transfunction.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/matterps.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/growthfactor.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/Downloads/matterps.py', wdir='C:/Users/kaderf/Downloads')
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/Compos/scripts/codes/matterps/script_mps.py', wdir='C:/Users/kaderf/Compos/scripts/codes/matterps')
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
x=3
print (x)
x=3j+1
print x
print (x)
type(x)
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/temp.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

c = StringIO("0 1 2 3 4 5 6 7 8 9\n0 1 4 9 16 25 36 49 64 81")
x,y=np.loadtxt(c)
X = np.linspace(0, 10, num=41, endpoint=True)
#y = x**2
#print (x,y)
f = interp1d(x, y, kind='cubic')
#xnew=np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x,y,'o',X,f(X))
plt.show()
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

c = StringIO("0 1 2 3 4 5 6 7 8 9\n0 1 4 9 16 25 36 49 64 81")
x,y=np.loadtxt(c)
X = np.linspace(0, 10, num=41, endpoint=True)
#y = x**2
#print (x,y)
f = interp1d(x, y, kind='cubic')
#xnew=np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x,y)
plt.show()
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

c = StringIO("0 1 2 3 4 5 6 7 8 9\n0 1 4 9 16 25 36 49 64 81")
x,y=np.loadtxt(c)
X = np.linspace(0, 10, num=41, endpoint=True)
#y = x**2
#print (x,y)
f = interp1d(x, y, kind='cubic')
#xnew=np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x,y,'o')
plt.show()
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

c = StringIO("0 1 3 5 6 8 9\n0 1 9 25 36 64 81")
x,y=np.loadtxt(c)
X = np.linspace(0, 10, num=41, endpoint=True)
#y = x**2
#print (x,y)
f = interp1d(x, y, kind='cubic')
#xnew=np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x,y,'o')
plt.show()
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

c = StringIO("0 1 3 5 6 8 9\n0 1 9 25 36 64 81")
x,y=np.loadtxt(c)
X = np.linspace(0, 10, num=41, endpoint=True)
print (max(x))
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

c = StringIO("0 1 3 5 6 8 9\n0 1 9 25 36 64 81")
x,y=np.loadtxt(c)
X = np.linspace(0, max(x), num=41, endpoint=True)
#y = x**2
#print (x,y)
f = interp1d(x, y, kind='cubic')
#xnew=np.linspace(0, 10, num=41, endpoint=True)
plt.plot(x,y,'o',X,f(X))
plt.show()
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
def f(x):
    f=x
    return f


def F(x):
    res = np.zeros_like(x)
    for i,val in enumerate(x):
        y,err = integrate.quad(f,0,val)
        res[i]=y
    return res


plt.plot(X,F(X))
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/func with var.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
import numpy
import scipy.integrate
import math

def w(r, theta, phi, alpha, beta, gamma):
    return(-math.log(theta * beta))

def integrand(phi, alpha, gamma, r, theta, beta):
    ww = w(r, theta, phi, alpha, beta, gamma)
    k = 1.
    T = 1.
    return (math.exp(-ww/(k*T)) - 1.)*r*r*math.sin(beta)*math.sin(theta)

# limits of integration

def zero(x, y=0):
    return 0.

def one(x, y=0):
    return 1.

def pi(x, y=0):
    return math.pi

def twopi(x, y=0):
    return 2.*math.pi

# integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
def secondIntegrals(r, theta, beta):
    res, err = scipy.integrate.tplquad(integrand, 0., 2.*math.pi, zero, twopi, zero, pi, args=(r, theta, beta))
    return res

# integrate over r [0, 1), beta [0, 2 Pi), theta [0, 2 Pi)
def integral():
    return scipy.integrate.tplquad(secondIntegrals, 0., 2.*math.pi, zero, twopi, zero, one)

expected = 16*math.pow(math.pi,5)/3.
result, err = integral()
diff = abs(result - expected)

print ("Result = ", result, " estimated error = ", err)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

l=np.linspace(0.,200.,1000)
def integrand(kpar, chi, chip):
    integrand=(kpar/(kpar**2+(1**2/chi**2)))**2*np.cos(kpar(chi-chip))/chi**2
    return integrand

# limits of integration

# integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
def zero(x, y=0):
    return 0.


def one(x, y=0):
    return 1.


def pi(x, y=0):
    return np.pi


def twopi(x, y=0):
    return 2.*np.pi


res, err = scipy.integrate.tplquad(integrand, 0., 1., zero, twopi, zero, one)

#d_Ls = np.linspace(0., 200., 1000)
 #   for index in range(1000):
  #      d_Ls[index] = d_L(zs[index], omega_m=omega_m)
# integrate over r [0, 1), beta [0, 2 Pi), theta [0, 2 Pi)
print ("Result = ", res)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

l=np.linspace(0.,200.,1000)
def integrand(kpar, chi, chip):
    integrand=(kpar/(kpar**2+(1/chi**2)))**2*np.cos(kpar*(chi-chip))/chi**2
    return integrand

# limits of integration

# integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
def zero(x, y=0):
    return 0.


def one(x, y=0):
    return 1.


def pi(x, y=0):
    return np.pi


def twopi(x, y=0):
    return 2.*np.pi


res, err = scipy.integrate.tplquad(integrand, 0., 1., zero, twopi, zero, one)
print ("Result = ", res)
for x in range(5):
    print 1+x
for x in range(5):
    print (1+x)
    
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

for l in range(10):
    def integrand(kpar, chi, chip,l):
        integrand=(kpar/(kpar**2+(l**2/chi**2)))**2*np.cos(kpar*(chi-chip))/chi**2
        return integrand
    # limits of integration
    
    # integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
    def zero(x, y=0):
        return 0.
    
    def one(x, y=0):
        return 1.
    
    def pi(x, y=0):
        return np.pi
    
    def twopi(x, y=0):
        return 2.*np.pi
    
    result, err = scipy.integrate.tplquad(integrand, 0., 1., zero, twopi, zero, one)
    print (result)

%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

for l in range(10):
    def integrand(kpar, chi, chip):
        integrand=(kpar/(kpar**2+(l**2/chi**2)))**2*np.cos(kpar*(chi-chip))/chi**2
        return integrand
    # limits of integration
    
    # integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
    def zero(x, y=0):
        return 0.
    
    def one(x, y=0):
        return 1.
    
    def pi(x, y=0):
        return np.pi
    
    def twopi(x, y=0):
        return 2.*np.pi
    
    result, err = scipy.integrate.tplquad(integrand, 0., 1., zero, twopi, zero, one)
    print (result)

%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

for l in range(10):
    def integrand(kpar, chi, chip):
        integrand=(kpar/(kpar**2+(l**2/chi**2)))**2*np.cos(kpar*(chi-chip))/chi**2
        return integrand
    # limits of integration
    
    # integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
    def zero(x, y=0):
        return 0.
    
    def one(x, y=0):
        return 1.
    
    def pi(x, y=0):
        return np.pi
    
    def twopi(x, y=0):
        return 2.*np.pi
    
    result, err = scipy.integrate.tplquad(integrand, 0., 1., zero, twopi, zero, one)
    print (result)

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

for l in range(10):
    def integrand(kpar, chi, chip):
        integrand=(kpar/(kpar**2+(l**2/chi**2)))**2*np.cos(kpar*(chi-chip))/chi**2
        return integrand
    # limits of integration
    
    # integrate over phi [0, Pi), alpha [0, 2 Pi), gamma [0, 2 Pi)
    def zero(x, y=0):
        return 0.
    
    def one(x, y=0):
        return 1.
    
    def pi(x, y=0):
        return np.pi
    
    def twopi(x, y=0):
        return 2.*np.pi
    
    result, err = scipy.integrate.tplquad(integrand, 0., 1., zero, twopi, zero, one)
    print (result)

%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt

x,y = np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float)
print (x)
runfile('C:/Users/kaderf/.spyder-py3/interpolation before integration.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x,y=np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                  unpack=True)
f = interp1d(x, y)
plt.plot(x,y,'o',x,f(x))
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x,y=np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                  unpack=True)
f = interp1d(x, y)
plt.plot(x,y,'o',x,f(x))
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

x,y=np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                  unpack=True)
f = interp1d(x, y)
plt.loglog(x,y,'o',x,f(x))
plt.show()
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
6./2.*3.

%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin ksz Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
d_co = cd.comoving_distance(6., **cosmo)
import cosmolopy.distance as cd
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/interpolation before integration.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Mon Jun 11 13:47:55 2018)---
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)/(np.sqrt(7)*np.pi)
print sp.trapz(sp.trapz(f, y[np.newaxis,:], axis=1), x, axis=0)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)/(np.sqrt(7)*np.pi)
print (sp.trapz(sp.trapz(f, y[np.newaxis,:], axis=1), x, axis=0))

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x**2 - y**2/7)/(np.sqrt(7)*np.pi)
print (sp.trapz(sp.trapz(f, y, axis=1), x, axis=0))


import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)/(np.sqrt(7)*np.pi)
print (sp.trapz(sp.trapz(f, y[np.newaxis,:], axis=1), x, axis=0))

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)/(np.sqrt(7)*np.pi)
print (sp.trapz(sp.trapz(f, y[np.newaxis,:]), x))

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)
z=np.linspace(-10, 10, 40)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z/(np.sqrt(7)*np.pi)
ff= (sp.trapz(sp.trapz(f, y[np.newaxis,:]), x))
fff=(sp.trapz(sp.trapz(ff, y[np.newaxis,:]), z))

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)
z=np.linspace(-10, 10, 80)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z/(np.sqrt(7)*np.pi)
ff= (sp.trapz(sp.trapz(f, y[np.newaxis,:]), x))
fff=(sp.trapz(sp.trapz(ff, y[np.newaxis,:]), z))

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)
z=np.linspace(-10, 10, 80)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z/(np.sqrt(7)*np.pi)
ff= (sp.trapz(sp.trapz(f, y[np.newaxis,:]), x))
fff=(sp.trapz(sp.trapz(ff, y[np.newaxis,:]), z))

x=np.arange(10)
x1=[,:5]
x1=[:,5]
x1=x[:,5]
x1=x[:,1]
x1=x[:,0]
import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)/(np.sqrt(7)*np.pi)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print sp.trapz(f, x, axis=0)
import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)

f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)/(np.sqrt(7)*np.pi)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (sp.trapz(f, x, axis=0))

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 200)
y = np.linspace(-10, 10, 80)
z=np.linspace(-10, 10, 80)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
ff= (sp.trapz(f, x, axis=0))
fff=sp.trapz(ff, z, axis=1)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 20)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)
fff=sp.trapz(ff, z, axis=1)
print (fff)

x=np.arange(5)
y=np.arange(4)
z=np.arange(6)
x
y
z
f = x[:,np.newaxis] +y[np.newaxis,:]+z[np.newaxis,:]
f = x[:,np.newaxis]*y[np.newaxis,:]*z[np.newaxis,:]
x=np.linspace(-10,10,200)
y=np.linspace(-10,10,80)
x.shape()
shape(x)
x
x[newaxis:,]
x[np.newaxis:,]
x=np.arange(4)
x
x.shape()
len(x)
xx=x[np.newaxis:,]
xx
x
x.shape
xx=x[np.newaxis:,]
xx
xx.shape
xx=x[np.newaxis,:]
xx.shape
x=np.linspace(5)
x=np.arange(5)
y=np.arange(2)
x[np.newaxis,:]
y[:,np.newaxis]
xx=x[np.newaxis,:]
yy=y[:,np.newaxis]
xx-yy
xy=xx-yy
xy.shape
y.shape
yy.shape
xx*yy
xx.shape
yy.shape
yy*xx
xx
yy
z=np.arange(6)
zz=z[:,np.newaxis]
zz
xx*yy*zz
(xx*yy).shape
z=np.arange(5)
zz=z[:,np.newaxis]
zz
xx*yy*zz
zz*xx*yy
xy=xx*yy
xy.shape
zz*xy
xy*zz
xy
zz
np.dot(xy,zz)
np.dot(zz,xy)
import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 20)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)
fff=sp.trapz(ff, z, axis=1)
print (fff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 20)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.dot(np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7),z[np.newaxis,:])/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)
fff=sp.trapz(ff, z, axis=1)
print (fff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 20)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.dot(z[np.newaxis,:],np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7))/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)
fff=sp.trapz(ff, z, axis=1)
print (fff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 8)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = z[np.newaxis,:]*np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7))/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)
fff=sp.trapz(ff, z, axis=1)
print (fff)
import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 8)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)
fff=sp.trapz(ff, z, axis=1)
print (fff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 8)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f)
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 8)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f),('first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 8)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f)
ff= (sp.trapz(f, x, axis=0))
print (ff)

import scipy as sp
import numpy as np
x = np.linspace(-10, 10, 8)
print (x)
y = np.linspace(-10, 10, 8)
print (y)
z=np.linspace(-10, 10, 8)
print (z)
f = np.exp(-x[:,np.newaxis]**2 - y[np.newaxis,:]**2/7)*z[np.newaxis,:]/(np.sqrt(7)*np.pi)
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
ff= (sp.trapz(f, x, axis=0))
print (ff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x*y*z
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
ff= (sp.trapz(f, x, axis=0))
print (ff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]*z[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
ff= (sp.trapz(f, x, axis=0))
print (ff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
ff= (sp.trapz(f, x, axis=0))
print (ff)


import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=np.dot(f,z[np.newaxis,:])
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]*z[np.newaxis,:])
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]*z[np.newaxis,:]
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]*z[np.newaxis,:]
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
fff= (sp.trapz(ff, x, axis=0))

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]*z[np.newaxis,:]
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
fff= (sp.trapz(ff, x, axis=0))
print fff
import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]*y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]*z[np.newaxis,:]
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
fff= (sp.trapz(ff, x, axis=0))
print (fff)

25**3
25**3/8
import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = x[:,np.newaxis]+y[np.newaxis,:]
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]+z[np.newaxis,:]
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
fff= (sp.trapz(ff, x, axis=0))
print (fff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = np.cos(x[:,np.newaxis]*y[np.newaxis,:])
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=np.cos(f[:,np.newaxis]*z[np.newaxis,:])
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
fff= (sp.trapz(ff, x, axis=0))
print (fff)

import scipy as sp
import numpy as np
x = np.arange(5)
print (x)
y = np.arange(5)
print (y)
z=np.arange(5)
print (z)
f = np.cos(x[:,np.newaxis]*y[np.newaxis,:])
print (f,'first f')
f=sp.trapz(f, y[np.newaxis,:], axis=1)
print (f,'second f')
f=f[:,np.newaxis]*np.cos(z[np.newaxis,:])
print (f,'third f')
ff= (sp.trapz(f, z, axis=1))
print (ff)
fff= (sp.trapz(ff, x, axis=0))
print (fff)

runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
x=1e3
type(x)
x=np.int(1e3)
x
type(x)
%clear
runfile('C:/Users/kaderf/.spyder-py3/trapz rule 3d.py', wdir='C:/Users/kaderf/.spyder-py3')
import CosmoloPy

import cosmolopy
import cosmolopy


runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Tue Jun 12 16:10:50 2018)---
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
x=np.arange(5)
y=np.arange(5)
z=np.arange(5)

runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Wed Jun 13 19:18:28 2018)---
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Wed Jun 13 21:33:13 2018)---
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
for z in range(10):
    chi = cd.comoving_distance(z, **cosmo)
    print ("Comoving distance to z is %.1f Mpc" % (chi))

%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import distance as cd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

#def zed():
z=np.arange(10)
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72}
for i in enumerate(z):
    chi = cd.comoving_distance(z, **cosmo)
    #print ("Comoving distance to z is %.1f Mpc" % (chi))

print (chi)
print (z)
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Sat Jun 16 14:20:10 2018)---
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import perturbation as cp
print (y(x))
f=cp.fgrowth(y(x), omega_M_0=0.27, unnormed=False)
print (f)
plt.loglog(y(x),f)

I=cp.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo)
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
import distance as cd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72}

def zed():
    z=np.arange(10)
    for i in enumerate(z):
        chi = cd.comoving_distance(z, **cosmo)
    f=interp1d(chi,z)
    return chi,f

x,y=zed()
    #print ("Comoving distance to z is %.1f Mpc" % (chi))
#print (chi)
#print (z)
#return res
#result=zed()
plt.loglog(x,y(x))
plt.show()
import perturbation as cp
print (y(x))
f=cp.fgrowth(y(x), omega_M_0=0.27, unnormed=False)
print (f)
plt.loglog(y(x),f)
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo)
import reionization as cr
I=cr.ionization_from_collapse(6, 1, 1e4, False, **cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 

## ---(Sun Jun 17 19:29:30 2018)---
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import perturbation as cp
print (y(x))
f=cp.fgrowth(y(x), omega_M_0=0.27, unnormed=False)
print (f)
plt.loglog(y(x),f)
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import perturbation as cp

fc = cp.collapse_fraction(*cp.sig_del(1e4, 0, **cosmo))
import perturbation as cp
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}

fc = cp.collapse_fraction(*cp.sig_del(1e4, 0, **cosmo))
import perturbation as cp
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}

#fc = cp.collapse_fraction(*cp.sig_del(1e4, 0, **cosmo))
fc=cp.collapse_fraction(sigma_min=1, delta_crit=2, sigma_mass=0, delta=0)
import perturbation as cp
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}

fc = cp.collapse_fraction(*cp.sig_del(1e4, 0, **cosmo))
import perturbation as cp
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}

fc = cp.collapse_fraction(*cp.sig_del(1e4, 0, **cosmo))
runfile('C:/Users/kaderf/.spyder-py3/distance.py', wdir='C:/Users/kaderf/.spyder-py3')
import perturbation as cp
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}

fc = cp.collapse_fraction(*cp.sig_del(1e4, 0, **cosmo))
runfile('C:/Users/kaderf/.spyder-py3/power.py', wdir='C:/Users/kaderf/.spyder-py3')
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
import reionization as cr
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : True}
I=cr.ionization_from_collapse(z=6, coeff_ion=1, temp_min=1e4, passed_min_mass = False,**cosmo) 
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
import distance as cd
cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0':0.0, 'h':0.72, 'omega_b_0' : 0.045, 'omega_n_0' : 0.0,
         'N_nu' : 0, 'n' : 1.0, 'sigma_8' : 0.9, 'baryonic_effects' : False}
         
chi = cd.comoving_distance(z=1100, **cosmo)
chi
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
from cosmolopy import b
print b
print (b)
print (b(chi))
print (chi,b(chi))
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/camb.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
xx=np.linspace(0,10,10)
xx
xxx=5
xx*xxx
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
36.*70**2
runfile('C:/Users/kaderf/.spyder-py3/constants.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/constants.py', wdir='C:/Users/kaderf/.spyder-py3')
70./3.*10**(19)
70./(3.*10**(19))
%clear
runfile('C:/Users/kaderf/.spyder-py3/constants.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/constants.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
0.3**0.53
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
1./3.*9.
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/constants.py', wdir='C:/Users/kaderf/.spyder-py3')
3*10**8*(3*10**22)**(-1)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/constants.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo, tau
import distance as cd
import density as den
import constants as cc

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
rho_c=3*H0**2/(8.*np.pi*cc.G_const_Mpc_Msun_s)
#rho_c=den.cosmo_densities(**cosmo)[0]
#print (rho_c)
gamma=0.53
omega_b_0=0.05/cosmo['h']**2
print (omega_b_0)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo, tau
import distance as cd
import density as den
import constants as cc

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
rho_c=3*H0**2/(8.*np.pi*cc.G_const_Mpc_Msun_s)
print (rho_c)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo, tau
import distance as cd
import density as den
import constants as cc

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
rho_c=3*H0**2/(8.*np.pi*cc.G_const_Mpc_Msun_s)
print (rho_c)
rho_c=den.cosmo_densities(**cosmo)[0]
print (rho_c)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo, tau
import distance as cd
import density as den
import constants as cc

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
rho_c=3*H0**2/(8.*np.pi*G)
print (rho_c)
rho_c=den.cosmo_densities(**cosmo)[0]
print (rho_c)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo, tau
import distance as cd
import density as den
import constants as cc

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
rho_c=3*H0**2/(8.*np.pi*G)
print (rho_c)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
print (rho_c)
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_Cl.py', wdir='C:/Users/kaderf/.spyder-py3')
class calc_funcs(object):

    def __init__(self):
        pass

    @staticmethod
    def add(a,b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b


calc_funcs(3,4)
add(3,4)
calc_funcs(add(3,4))
calc_funcs().add(3,4)
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
0.0001
1e-4
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Mon Jun 25 12:19:17 2018)---
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Mon Jun 25 14:11:01 2018)---
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
x=0
np.logx
np.log(x)
x=1
np.log(x)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
np.log(1100)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
np.log(1)
np.log(0.1)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
z=np.logspace(np.log(1),np.log(1100),1101)
z
%clear
z1=np.linspace(0,1100,1101)
z1
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
chi_m = cd.comoving_distance(1100, **cosmo)
chi_m
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
chi_m = cd.comoving_distance(1, **cosmo)
chi_m
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
chi_full = cd.comoving_distance(0, **cosmo)
chi_full
chi_full = cd.comoving_distance(1, **cosmo)
chi_full
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
np.linspace(1,10,10)
np.logspace(0,1,10)
np.linspace(-1,10,12)
np.log(1)
np.log(-6)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/reionization.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
def f(x):
    f1=x**2
    f2=2*x
    return f1,f2

x=2
f(x)
f(x)[0]
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
@author: KaderF
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 20:01:45 2018

@author: KaderF
"""

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2

plt.plot(z,tau_der_1)
plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2

plt.plot(z,tau_der_1)
plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2

print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
plt.plot(z,tau_der_1)
plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2

print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
plt.plot(z,tau_der_1)
plt.plot(z,tau_der_2)
plt.show()
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
cc.c_light_Mpc_s
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

'''
tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2

print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
plt.plot(z,tau_der_1)
plt.plot(z,tau_der_2)
plt.show()
'''

def g(chi):
    f=(den.omega_M_z(z_r,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=(f*H0*2*np.pi)*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    return g


chi=np.linspace(0,np.inf,1000)
g_norm_integral=sp.integrate.trapz(g(chi),chi)
print (g_norm_integral)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)


tau=[]
tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau

plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
x=np.linspace(0,np.inf,1000)
x=np.linspace(0,np.inf,np.inf)
from scipy import integrate
def f(x):
    f=np.exp(-x)
    return f

integrate.quad(f(x), 0, np.inf)
integrate.quad(f, 0, np.inf)
import scipy as sp
sp.integrate.integrate.quad
sp.integrate.quad
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

'''
tau=[]
tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau

plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
'''

def g(chi):
    f=(den.omega_M_z(z_r,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=(f*H0*2*np.pi)*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    return g


integral_gnorm=sp.integrate.quad(g,0,np.inf)
print (integral_gnorm)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)


tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.plot(z,f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)


tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.plot(f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)


tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z_r,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.plot(f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

f=(den.omega_M_z(z_r,**cosmo))**(gamma)
print (f)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.plot(f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.plot(f)
print (f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.loglog(f)
print (f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()
import density as den
z=np.linspace(0,1100,1101)
cosmo={'omega_M_0':0.3}
f=(den.omega_M_z(z,**cosmo))**(gamma)
cosmo={'omega_M_0':0.3,'omega_lambda_0':0.7}
f=(den.omega_M_z(z,**cosmo))**(gamma)
f
plt.plot(f)
plt.plot(den.omega_M_z(z,**cosmo))
z
omega_M=[]
for i in z:
    om=den.omega_M_z(z,**cosmo)
    omega_M+=om
    
for i in z:
    omega_M=den.omega_M_z(z,**cosmo)
    omega_M+=om
    
for i in z:
    omega_M=den.omega_M_z(z,**cosmo)
    omega_M+=om
    
z
omega_M=[]
cosmo
for i in z:
    omega_M=den.omega_M_z(z,**cosmo)
    omega_M+=omega_M
    
plt.plot(omega_M)
x=np.linspace(0,10,10)
y=x**2
plt.plot(x,y)
cosmo={'omega_M_0':0.3,'omega_lambda_0':0.7,'omega_k_0':0.0}
z
omega_M=[]
cosmo
for i in z:
    omega_M=den.omega_M_z(z,**cosmo)
    omega_M+=omega_M
    
plt.plot(omega_M)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)
'''
tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.loglog(f)
print (f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()

'''
def g(chi):
    f=(den.omega_M_z(z_r,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=(f*H0*2*np.pi)*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    return g


g_norm=sp.integrate.quad(g,0,np.inf)
print (g_norm)
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr

#print (chi_m)

n_points=200
H0=cc.H100_s*cosmo['h']
z_r=20.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
chi_r = cd.comoving_distance(z_r, **cosmo)
chi_m = cd.comoving_distance(1100, **cosmo)
#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]
tau_r=0.5

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)
'''
tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.loglog(f)
print (f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()

'''
def g(chi):
    f=(den.omega_M_z(z_r,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=(f*H0*2*np.pi)*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    g_int = sp.interpolate.interp1d(chi,g(chi))
    return g_int


g_norm=sp.integrate.quad(g,0,np.inf)
print (g_norm)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
print (cc.c_light_Mpc_s)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
a=Mps_div_kpar_sq[0::100]
a
sum(a)
a[0]
a[19]
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
np.log(2)
np.log(1)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
cc.c_light_Mpc_s
chi_r
chi_m
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/density.py', wdir='C:/Users/kaderf/.spyder-py3')
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
import density as den
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
%clear
runfile('C:/Users/kaderf/.spyder-py3/cosmolopy.py', wdir='C:/Users/kaderf/.spyder-py3')
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
import density as den
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from cosmolopy import zed, cosmo
import distance as cd
import density as den
import constants as cc
import reionization as cr
from decimal import Decimal

#print (chi_m)

n_points=2000
H0=cc.H100_s*cosmo['h']
z_r=6.0
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
print (rho_c)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
1./1.7/0.3
1.0 / (1. + (1. - cosmo['omega_M_0'])/ (cosmo['omega_M_0'] * (1. + z)**3.))
1.0 / (1. + (1. - cosmo['omega_M_0'])/ (cosmo['omega_M_0'] * (1. + 0)**3.))
1.0 / (1. + (1. - cosmo['omega_M_0']))
1./1.7
0.58/0.3
0.58*0.3
(cosmo['omega_M_0'] * (1. + 0)**3.))
(cosmo['omega_M_0'] * (1. + 0)**3.)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
0.3*9
0.3*27.
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
0.045/0.3*100
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
def f(x):
    y=3*x
    return y

f(3)
type(f(3))
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
x=5
x.shape
x.shape(
)
shape(x)
len(x)
x=5.0
len(x)
shape(x)
x.shape()
dim(x)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
ell=np.linspace(0,10,10)
type(ell)
shape(ell)
ell.shape()
ell.shape
x.shape
x=5
x.shape
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
print (chi(1))
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
D_1(1)
f(1)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
x=np.linspace(0,10,10)
y=x**(-4)
x=np.linspace(1,10,10)
y=x**(-4)
plt.plot(x,y)
x=np.linspace(1,10,100)
plt.plot(x,y)
y=x**(-4)
plt.plot(x,y)
y=x**(-2)
plt.plot(x,y)
y=y+1
plt.plot(x,y)
y=y-1
plt.plot(x,y)
y=(1+x)**(-2)
plt.plot(x,y)
y=(10+x)**(-2)
plt.plot(x,y)
y=(100+x)**(-2)
plt.plot(x,y)
y=(1000+x)**(-2)
plt.plot(x,y)
y=(1e8+x)**(-2)
plt.plot(x,y)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
H(1)
r(1)
f(1)
D(1)
D_1(1)
chi(1)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
Mps_zero_ell
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
H0
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
T_mean(2)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
type(lksz.f)
type(uf.H0)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
y=2
f(z_r)
lksz.f(z_r)
lksz.f
uf.f(z_r)
uf.f(z=1)
uf.f(1100)
H0
uf.H0
cc.c_light_Mpc_s
uf.T_mean(z_r)
uf.D_1(z_r)
uf.r(z_r)
lksz.g(chi_r)
kpar=y/uf.r(z_r)
kpar
uf.Mps_interpf(kpar)
uf.r(1)
y
y/uf.r(1)
kpar_1=y/uf.r(1)
uf.Mps_interpf(kpar_1)
kpar_r=2./r(z_r)
kpar_r=2./uf.r(z_r)
kpar
uf.Mps_interpf(kpar_r)
kpar_r
uf.Mps_interpf(kpar_r)
kperp=1./uf.chi_r
kperp=1./uf.chi(z_r)
kperp
k=np.sqrt(kpar_r**2+kperp**2)
k
uf.Mps_interpf(k)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
uf.Mps_interpf(kpar)
1e-10+1e-8
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
uf.H0/cc.c_light_Mpc_s
uf.Mps_interpf(kpar)
kpar_r=2./r(z_r)
kpar_r=2./uf.r(z_r)
uf.Mps_interpf(kpar)
uf.Mps_interpf(kpar_r)
kperp=1./uf.chi(z_r)
k=np.sqrt(kpar_r**2+kperp**2)
uf.Mps_interpf(k)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
uf.T_mean(z_r)
a
kpar=y/uf.r(z_r)
y=2
kpar=y/uf.r(z_r)
kpar
k=np.sqrt(kpar**2+(1./uf.chi(z_r))**2)
k
mu_k=kpar/k
mu_k
a=uf.b_HI+lksz.f*mu_k**2
a
0.6**2
uf.T_mean(z_r)*a*uf.D_1(z_r)
uf.T_mean(z_r)
uf.D_1(z_r)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
g(chi_r)
g(chi_r)
chi_r=uf.chi(6)
g(chi_r)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import density as den
import constants as cc
import reionization as cr
import useful_functions as uf
#print (chi_m)

cosmo=uf.cosmo
zed=uf.zed
n_points=200
H0=cc.H100_s*cosmo['h']
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
z_r=6.
chi_r=uf.chi(z_r)
chi_m = uf.chi(1100)
chi=np.linspace(0.0001,chi_m,n_points)

#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)
'''
tau=[]
tau_der_1=[]
tau_der_2=[]
f=[]
z=np.linspace(0,1100,1101)
for i in z:
    f=(den.omega_M_z(z,**cosmo))**(gamma)
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau
    f+=f

plt.loglog(f)
print (f)
#plt.plot(tau)
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
#plt.plot(z,tau_der_1)
#plt.plot(z,tau_der_2)
plt.show()

'''
def g(chi):
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=(uf.f(zed(chi))*H0*2*np.pi)*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    return g

plt.loglog(chi,g(chi))
plt.show()
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import density as den
import constants as cc
import reionization as cr
import useful_functions as uf
#print (chi_m)

cosmo=uf.cosmo
zed=uf.zed
n_points=200
H0=cc.H100_s*cosmo['h']
x_e=1.
G=cc.G_const_Mpc_Msun_s/cc.M_sun_g
#rho_c=3*H0**2/(8.*np.pi*G)
rho_c=den.cosmo_densities(**cosmo)[0]*cc.M_sun_g
gamma=0.53
sigma_T=cc.sigma_T_Mpc
z_r=6.
chi_r=uf.chi(z_r)
chi_m = uf.chi(1100)
chi=np.linspace(0.0001,chi_m,n_points)

#print (omega_b_0)
#tau_r=0.046*omega_b_0*h*x_e[np.sqrt(omega_M_0(1+z_r)**3+(1-omega_M_0))-1]

#get matter power spec data
kabs,P= np.genfromtxt('C:\\Users\\kaderf\\Downloads\\camb_84189500_matterpower_z0.dat', dtype=float,
                      unpack=True)

#interpolate the matter power spec
Mps_interpf = sp.interpolate.interp1d(kabs, P, bounds_error=False,fill_value=0.)

tau=[]
tau_der_1=[]
tau_der_2=[]
z=np.linspace(0,1100,1101)
for i in z:
    tau=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    tau_der_1=cr.integrate_optical_depth(z, x_ionH=1.0, x_ionHe=2.0, **cosmo)[0]
    tau_der_1+=tau_der_1
    tau_der_2=(1.+z)**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    tau_der_2+=tau_der_2
    tau+=tau

plt.plot(tau)
plt.show()
#print(max(np.abs(tau_der_2))-max(np.abs(tau_der_1)))
plt.plot(z,tau_der_1)
plt.plot(z,tau_der_2)
plt.show()


def g(chi):
    tau=cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
    #tau_der=np.abs(cr.integrate_optical_depth(zed(chi), x_ionH=1.0, x_ionHe=1.0, **cosmo))[0]
    tau_der=(1.+zed(chi))**2*cosmo['omega_b_0']*rho_c*x_e*sigma_T/cc.m_p_g
    g=(uf.f(zed(chi))*H0*2*np.pi)*tau_der*np.exp(-tau)/cc.c_light_Mpc_s
    return g

plt.loglog(chi,g(chi))
plt.show()
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
uf.chi(6)
0.1e-7
g(1000)
g(15000)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
g(15000)
tau=cr.integrate_optical_depth(6, x_ionH=1.0, x_ionHe=1.0, **cosmo)[1]
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
g_int(6)
g_int(15000)
g_int(uf.chi(6))
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
g_int(uf.chi(6))
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/twentyonecm_autocorr.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
f(0)
f(-1)
f(-2)
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
den.omega_M_z(1100,**cosmo)
den.omega_M_z(110,**cosmo)
den.omega_M_z(0,**cosmo)
den.omega_M_z(0.0001,**cosmo)
den.omega_M_z(1e-10,**cosmo)
den.omega_M_z(1e-20,**cosmo)
plt.plot(x,x**0.53)
x=np.linspace(0,10,10)
plt.plot(x,x**0.53)
plt.ylim(0,1)
plt.plot(x,x**0.53)
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
uf.f(z)
uf.f(0)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
uf.kpar(10,1)
uf.r(1)
uf.H(1)
cc.c_light_Mpc_s
kperp=1/uf.chi(1)
kperp
uf.chi(1)
k=np.sqrt(uf.kpar(10,1)**2+kperp**2)
k
mu_k=uf.kpar(10,1)/k
mu_k
uf.f(1)
a=1+uf.f(1)*mu_k**2
a
uf.T(1)
uf.T_mean(1)
uf.omega_HI
H0
uf.H(0)
cc.H100_s
uf.H(1)
0.0008/0.003
ell=np.linspace(0,100,100)
Cl_21_lksz(ell,y=10,z=1)
a
D_1(1)
uf.D_1(1)
uf.chi(1)
uf.r(1)
uf.T_mean(z)*a*uf.D_1(z))/(uf.chi(z)**2*uf.r(z)
(uf.T_mean(z)*a*uf.D_1(z))/(uf.chi(z)**2*uf.r(z))
z=1
(uf.T_mean(z)*a*uf.D_1(z))/(uf.chi(z)**2*uf.r(z))
np.sin(uf.kpar(y,z_r)*chi_r)
y=10
np.sin(uf.kpar(y,z_r)*chi_r)
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
print (Cl_21_lksz(ell,y=1,z=1))
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/21cm_linksz_cross_corr.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear

## ---(Tue Jul  3 14:18:41 2018)---
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')

## ---(Tue Jul  3 19:48:17 2018)---
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/useful_functions.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
chi_m
chi_r
chi_m-chi_r
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_method2.py', wdir='C:/Users/kaderf/.spyder-py3')
chi_m-chi_r
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')
%clear
runfile('C:/Users/kaderf/.spyder-py3/lin_ksz_auto_method1.py', wdir='C:/Users/kaderf/.spyder-py3')                                                                                                                                           