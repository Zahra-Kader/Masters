# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 02:19:59 2018

@author: KaderF
"""
#If you want to get the integral of x**3 and you have the functions x and x**2, 
#then its easy. Just multiply the two since x*x**2=x**3. But what if you have the
#function x, and a text file with x and y values. You take the textfile and load it.
#Interpolate the values in the textfile so that you have the function y=f(x) where in 
#this case f(x)=x**2. NOW you have x and f(x)=x**2 where the latter was found by 
#interpolating a text file. NOW we can multiply the two (to get x**3) and integrate!!! 

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from io import StringIO   # StringIO behaves like a file object

#load the textfile c, c isnt a textfile but it acts as one in this case
c = StringIO("0 1 2 3 4 5 6 7 8 9\n0 1 4 9 16 25 36 49 64 81")
x,y=np.loadtxt(c)

#interpolate
X = np.linspace(0, max(x), num=41, endpoint=True)
#y = x**2
#print (x,y)
f = interp1d(x, y, kind='cubic')
plt.plot(x,y,'o',X,f(X))
plt.show()

#define new function that we want to integrate
def f2(x):
    f2=x*f(x)
    return f2

#plot the integral
def F(x):
    res = np.zeros_like(x)
    for i,val in enumerate(x):
        y,err = integrate.quad(f2,0,val)
        res[i]=y
    return res

#plot the resulting integral and check that it is indeed the integral of x**3, which 
#is x**4/4
plt.plot(X,F(X),X,X**4/4.0)
#Looking at the plots we see that the results are good!
#Note: Can also plot f(x) on x axis